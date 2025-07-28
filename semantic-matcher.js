// semantic-matcher.js - Advanced Semantic Matching Engine
import { cosineSimilarity, fuzzyMatch } from "./matching_utils.js";

export class SemanticMatcher {
  constructor(ollamaHandler = null, config = {}) {
    this.ollamaHandler = ollamaHandler;
    this.config = {
      minMatchScore: config.minMatchScore || 0.1,
      aiEnhancedMode: config.aiEnhancedMode !== false, // Default to true if Ollama available
      fuzzyWeight: config.fuzzyWeight || 0.3,
      cosineWeight: config.cosineWeight || 0.4,
      aiWeight: config.aiWeight || 0.3,
      maxConcurrentAIRequests: config.maxConcurrentAIRequests || 3,
      timeoutMs: config.timeoutMs || 30000,
    };

    this.stats = {
      totalMatches: 0,
      aiEnhancedMatches: 0,
      traditionalMatches: 0,
      failedAIMatches: 0,
    };
  }

  /**
   * Find best matches between topics and document content
   * @param {Array} topics - Array of curated topics
   * @param {object} documentStructure - Document structure from analyzer
   * @param {string} filename - Document filename
   * @returns {Promise<Array>} - Array of match objects
   */
  async findBestMatches(topics, documentStructure, filename) {
    console.log(
      `🔍 [Semantic Matcher] Finding matches for ${topics.length} topics in ${filename}`
    );

    const matches = [];
    const searchIndex = this._createSearchIndex(documentStructure);

    // Process topics with controlled concurrency for AI requests
    const topicBatches = this._createBatches(
      topics,
      this.config.maxConcurrentAIRequests
    );

    for (const batch of topicBatches) {
      const batchPromises = batch.map((topic, batchIndex) =>
        this._findTopicMatches(
          topic,
          searchIndex,
          filename,
          topics.indexOf(topic)
        )
      );

      const batchResults = await Promise.all(batchPromises);
      matches.push(...batchResults.flat());
    }

    // Sort matches by importance rank and score
    matches.sort((a, b) => {
      if (a.importance_rank !== b.importance_rank) {
        return a.importance_rank - b.importance_rank;
      }
      return b.match_score - a.match_score;
    });

    console.log(
      `✅ [Semantic Matcher] Found ${matches.length} matches for ${filename}`
    );
    this._logStats();

    return matches;
  }

  /**
   * Find matches for a single topic
   * @param {string} topic - Topic to match
   * @param {Array} searchIndex - Document search index
   * @param {string} filename - Document filename
   * @param {number} topicIndex - Topic index for ranking
   * @returns {Promise<Array>} - Array of matches for this topic
   */
  async _findTopicMatches(topic, searchIndex, filename, topicIndex) {
    const candidates = [];

    // Traditional matching for all searchable elements
    for (const element of searchIndex) {
      const traditionalScore = this._calculateTraditionalScore(
        topic,
        element.text
      );

      if (traditionalScore > 0) {
        candidates.push({
          element,
          traditionalScore,
          aiScore: 0, // Will be filled later if AI is available
        });
      }
    }

    // Sort candidates by traditional score and keep top candidates for AI enhancement
    candidates.sort((a, b) => b.traditionalScore - a.traditionalScore);
    const topCandidates = candidates.slice(0, 5); // Limit AI requests

    // AI-enhanced scoring for top candidates
    if (
      this.ollamaHandler &&
      this.config.aiEnhancedMode &&
      topCandidates.length > 0
    ) {
      try {
        const aiPromises = topCandidates.map(async (candidate) => {
          try {
            const aiScore = await this.ollamaHandler.getSemanticRelevance(
              topic,
              candidate.element.text
            );
            candidate.aiScore = aiScore;
            this.stats.aiEnhancedMatches++;
            return candidate;
          } catch (error) {
            console.warn(
              `⚠️ [Semantic Matcher] AI scoring failed for topic "${topic}":`,
              error.message
            );
            this.stats.failedAIMatches++;
            return candidate; // Keep traditional score
          }
        });

        await Promise.all(aiPromises);
      } catch (error) {
        console.warn(
          `⚠️ [Semantic Matcher] AI enhancement batch failed:`,
          error.message
        );
      }
    }

    // Calculate final scores and create matches
    const matches = [];
    for (const candidate of candidates) {
      const finalScore = this._calculateFinalScore(
        candidate.traditionalScore,
        candidate.aiScore
      );

      if (finalScore >= this.config.minMatchScore) {
        matches.push({
          document: filename,
          topic: topic,
          section_title: this._truncateText(candidate.element.text, 100),
          page_number: candidate.element.page,
          importance_rank: topicIndex + 1,
          match_score: parseFloat(finalScore.toFixed(3)),
          match_type: candidate.element.type,
          traditional_score: parseFloat(candidate.traditionalScore.toFixed(3)),
          ai_score: parseFloat(candidate.aiScore.toFixed(3)),
          element_importance: candidate.element.importance,
          bbox: candidate.element.bbox || null,
        });
      }
    }

    this.stats.totalMatches += matches.length;
    if (!this.ollamaHandler || !this.config.aiEnhancedMode) {
      this.stats.traditionalMatches += matches.length;
    }

    return matches;
  }

  /**
   * Calculate traditional matching score using fuzzy + cosine similarity
   * @param {string} topic - Topic text
   * @param {string} text - Content text
   * @returns {number} - Traditional score (0-1)
   */
  _calculateTraditionalScore(topic, text) {
    if (!text || text.length < 3) return 0;

    const fuzzyScore = fuzzyMatch(topic, text);
    const cosineScore = cosineSimilarity(topic, text);

    return (
      fuzzyScore * this.config.fuzzyWeight +
      cosineScore * this.config.cosineWeight
    );
  }

  /**
   * Calculate final combined score
   * @param {number} traditionalScore - Traditional matching score
   * @param {number} aiScore - AI-enhanced score
   * @returns {number} - Final combined score
   */
  _calculateFinalScore(traditionalScore, aiScore) {
    if (!this.ollamaHandler || !this.config.aiEnhancedMode || aiScore === 0) {
      // No AI enhancement, normalize traditional score
      return (
        traditionalScore / (this.config.fuzzyWeight + this.config.cosineWeight)
      );
    }

    // Combine traditional and AI scores with weights
    const normalizedTraditional =
      traditionalScore / (this.config.fuzzyWeight + this.config.cosineWeight);
    return (
      normalizedTraditional * (1 - this.config.aiWeight) +
      aiScore * this.config.aiWeight
    );
  }

  /**
   * Create searchable index from document structure
   * @param {object} documentStructure - Document structure
   * @returns {Array} - Search index
   */
  _createSearchIndex(documentStructure) {
    const searchIndex = [];

    // Add document title
    if (documentStructure.title) {
      searchIndex.push({
        type: "title",
        text: documentStructure.title,
        page: 1,
        section: null,
        importance: 1.0,
        bbox: null,
      });
    }

    // Add sections and content
    for (const section of documentStructure.sections) {
      // Add section header
      searchIndex.push({
        type: "section-header",
        text: section.title,
        page: section.page,
        section: section.index,
        importance: 0.8,
        bbox: section.bbox || null,
      });

      // Add section content
      for (const content of section.content) {
        if (content.text && content.text.length > 10) {
          searchIndex.push({
            type: content.type.toLowerCase().replace("-", "_"),
            text: content.text,
            page: content.page,
            section: section.index,
            importance: this._getContentImportance(content.type),
            bbox: content.bbox || null,
          });
        }
      }
    }

    return searchIndex;
  }

  /**
   * Get content importance score based on type
   * @param {string} contentType - Type of content
   * @returns {number} - Importance score
   */
  _getContentImportance(contentType) {
    const importanceMap = {
      Title: 1.0,
      "Section-header": 0.8,
      Text: 0.6,
      "List-item": 0.5,
      Caption: 0.4,
      Table: 0.7,
      Formula: 0.3,
    };

    return importanceMap[contentType] || 0.4;
  }

  /**
   * Create batches for controlled concurrency
   * @param {Array} items - Items to batch
   * @param {number} batchSize - Size of each batch
   * @returns {Array} - Array of batches
   */
  _createBatches(items, batchSize) {
    const batches = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push(items.slice(i, i + batchSize));
    }
    return batches;
  }

  /**
   * Truncate text to specified length
   * @param {string} text - Text to truncate
   * @param {number} maxLength - Maximum length
   * @returns {string} - Truncated text
   */
  _truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength - 3) + "...";
  }

  /**
   * Generate refined analysis from matches
   * @param {Array} matches - Array of match objects
   * @returns {Array} - Refined analysis
   */
  generateRefinedAnalysis(matches) {
    return matches.map((match) => ({
      document: match.document,
      page_number: match.page_number,
      refined_text: `${this._getMatchTypeLabel(match.match_type)}: ${
        match.section_title
      }`,
      topic_matched: match.topic,
      match_score: match.match_score,
      importance_rank: match.importance_rank,
      traditional_score: match.traditional_score,
      ai_enhanced: match.ai_score > 0,
      ai_score: match.ai_score,
      confidence_level: this._getConfidenceLevel(match.match_score),
    }));
  }

  /**
   * Get human-readable match type label
   * @param {string} matchType - Match type
   * @returns {string} - Human-readable label
   */
  _getMatchTypeLabel(matchType) {
    const labelMap = {
      title: "Document Title",
      "section-header": "Section Header",
      section_header: "Section Header",
      text: "Content",
      "list-item": "List Item",
      list_item: "List Item",
      caption: "Caption",
      table: "Table",
      formula: "Formula",
    };

    return labelMap[matchType] || "Content";
  }

  /**
   * Get confidence level based on match score
   * @param {number} score - Match score
   * @returns {string} - Confidence level
   */
  _getConfidenceLevel(score) {
    if (score >= 0.8) return "High";
    if (score >= 0.5) return "Medium";
    if (score >= 0.3) return "Low";
    return "Very Low";
  }

  /**
   * Log matching statistics
   */
  _logStats() {
    console.log(`📊 [Semantic Matcher] Statistics:`);
    console.log(`   Total matches: ${this.stats.totalMatches}`);
    console.log(`   AI-enhanced: ${this.stats.aiEnhancedMatches}`);
    console.log(`   Traditional only: ${this.stats.traditionalMatches}`);
    if (this.stats.failedAIMatches > 0) {
      console.log(`   Failed AI requests: ${this.stats.failedAIMatches}`);
    }
  }

  /**
   * Reset statistics
   */
  resetStats() {
    this.stats = {
      totalMatches: 0,
      aiEnhancedMatches: 0,
      traditionalMatches: 0,
      failedAIMatches: 0,
    };
  }

  /**
   * Get current statistics
   * @returns {object} - Current statistics
   */
  getStats() {
    return { ...this.stats };
  }

  /**
   * Update configuration
   * @param {object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    console.log(`🔧 [Semantic Matcher] Configuration updated:`, newConfig);
  }

  /**
   * Test semantic matching with sample data
   * @param {string} topic - Test topic
   * @param {string} text - Test text
   * @returns {Promise<object>} - Test results
   */
  async testMatching(topic, text) {
    const traditionalScore = this._calculateTraditionalScore(topic, text);
    let aiScore = 0;

    if (this.ollamaHandler && this.config.aiEnhancedMode) {
      try {
        aiScore = await this.ollamaHandler.getSemanticRelevance(topic, text);
      } catch (error) {
        console.warn("AI scoring test failed:", error.message);
      }
    }

    const finalScore = this._calculateFinalScore(traditionalScore, aiScore);

    return {
      topic,
      text: this._truncateText(text, 100),
      traditional_score: parseFloat(traditionalScore.toFixed(3)),
      ai_score: parseFloat(aiScore.toFixed(3)),
      final_score: parseFloat(finalScore.toFixed(3)),
      confidence_level: this._getConfidenceLevel(finalScore),
      ai_enhanced: aiScore > 0,
    };
  }
}

export default SemanticMatcher;
