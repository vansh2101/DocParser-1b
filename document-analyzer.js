// document-analyzer.js - Document Structure Analysis and Text Extraction
import fs from "fs-extra";
import path from "path";

export class DocumentAnalyzer {
  constructor(config = {}) {
    this.minTextLength = config.minTextLength || 10;
    this.sectionThreshold = config.sectionThreshold || 0.7; // Confidence threshold for section headers
    this.titleThreshold = config.titleThreshold || 0.8; // Confidence threshold for titles
  }

  /**
   * Create hierarchical document structure from page results
   * @param {Array} pageResults - Results from PDF processor
   * @returns {object} - Document structure and extracted text
   */
  createDocumentStructure(pageResults) {
    const structure = {
      title: null,
      sections: [],
      pages: [],
      metadata: {
        totalElements: 0,
        textElements: 0,
        sectionHeaders: 0,
        titles: 0,
      },
    };

    let currentSection = null;
    let allText = "";
    let allDetections = [];

    for (const pageResult of pageResults) {
      const pageContent = {
        pageNumber: pageResult.pageNumber,
        elements: [],
        metadata: {
          processingTime: pageResult.processingTime,
          totalElements: pageResult.detections.length,
        },
      };

      // Sort detections by reading order for proper text flow
      const sortedDetections = [...pageResult.detections].sort((a, b) => {
        const yDiff = a.center[1] - b.center[1];
        if (Math.abs(yDiff) > 20) return yDiff;
        return a.center[0] - b.center[0];
      });

      for (const detection of sortedDetections) {
        const element = {
          type: detection.label,
          text: detection.extractedText || "",
          confidence: detection.confidence,
          bbox: detection.bbox,
          normalizedBbox: detection.normalizedBbox,
          reading_order: detection.reading_order,
          area: detection.area,
          center: detection.center,
        };

        pageContent.elements.push(element);
        allDetections.push(detection);

        // Update metadata counters
        structure.metadata.totalElements++;
        if (detection.label === "Title") structure.metadata.titles++;
        if (detection.label === "Section-header")
          structure.metadata.sectionHeaders++;
        if (["Text", "List-item"].includes(detection.label))
          structure.metadata.textElements++;

        // Build hierarchical structure
        if (
          detection.label === "Title" &&
          !structure.title &&
          detection.confidence >= this.titleThreshold &&
          detection.extractedText
        ) {
          structure.title = this._cleanText(detection.extractedText);
          console.log(
            `📑 [Document Analyzer] Found document title: "${structure.title}"`
          );
        } else if (
          detection.label === "Section-header" &&
          detection.confidence >= this.sectionThreshold &&
          detection.extractedText
        ) {
          // Finalize current section
          if (currentSection && currentSection.content.length > 0) {
            structure.sections.push(currentSection);
          }

          // Start new section
          currentSection = {
            title: this._cleanText(detection.extractedText),
            page: pageResult.pageNumber,
            confidence: detection.confidence,
            bbox: detection.bbox,
            content: [],
            metadata: {
              wordCount: 0,
              elementCount: 0,
            },
          };

          console.log(
            `📂 [Document Analyzer] New section: "${currentSection.title}" (Page ${pageResult.pageNumber})`
          );
        } else if (
          currentSection &&
          ["Text", "List-item", "Caption"].includes(detection.label) &&
          detection.extractedText
        ) {
          const cleanText = this._cleanText(detection.extractedText);
          if (cleanText.length >= this.minTextLength) {
            currentSection.content.push({
              type: detection.label,
              text: cleanText,
              page: pageResult.pageNumber,
              confidence: detection.confidence,
              bbox: detection.bbox,
            });

            currentSection.metadata.wordCount += cleanText.split(/\s+/).length;
            currentSection.metadata.elementCount++;
          }
        }

        // Accumulate all meaningful text for summarization
        if (
          detection.extractedText &&
          detection.extractedText.length >= this.minTextLength &&
          !detection.extractedText.includes("OCR not available") &&
          !detection.extractedText.includes("extraction failed")
        ) {
          const cleanText = this._cleanText(detection.extractedText);
          if (cleanText.length >= this.minTextLength) {
            allText += cleanText + "\n";
          }
        }
      }

      structure.pages.push(pageContent);
    }

    // Add final section if exists
    if (currentSection && currentSection.content.length > 0) {
      structure.sections.push(currentSection);
    }

    // Post-process structure
    this._postProcessStructure(structure);

    console.log(
      `✅ [Document Analyzer] Structure created: ${structure.sections.length} sections, ${structure.metadata.totalElements} elements`
    );

    return {
      structure,
      allText: allText.trim(),
      allDetections,
      statistics: this._generateStatistics(structure, allText),
    };
  }

  /**
   * Post-process document structure for better organization
   * @param {object} structure - Document structure to process
   */
  _postProcessStructure(structure) {
    // Remove empty sections
    structure.sections = structure.sections.filter(
      (section) => section.content.length > 0 || section.title.length > 3
    );

    // Sort sections by page number
    structure.sections.sort((a, b) => a.page - b.page);

    // If no title found, try to infer from first section or filename
    if (!structure.title && structure.sections.length > 0) {
      const firstSection = structure.sections[0];
      if (firstSection.title.length > 5) {
        structure.title = firstSection.title;
        console.log(
          `📑 [Document Analyzer] Inferred title from first section: "${structure.title}"`
        );
      }
    }

    // Add section indices
    structure.sections.forEach((section, index) => {
      section.index = index + 1;
    });
  }

  /**
   * Generate document statistics
   * @param {object} structure - Document structure
   * @param {string} allText - All extracted text
   * @returns {object} - Statistics object
   */
  _generateStatistics(structure, allText) {
    const words = allText.split(/\s+/).filter((word) => word.length > 0);
    const sentences = allText
      .split(/[.!?]+/)
      .filter((s) => s.trim().length > 0);

    return {
      totalPages: structure.pages.length,
      totalSections: structure.sections.length,
      totalElements: structure.metadata.totalElements,
      textElements: structure.metadata.textElements,
      sectionHeaders: structure.metadata.sectionHeaders,
      titles: structure.metadata.titles,
      wordCount: words.length,
      sentenceCount: sentences.length,
      averageWordsPerSentence:
        sentences.length > 0 ? Math.round(words.length / sentences.length) : 0,
      sectionsWithContent: structure.sections.filter(
        (s) => s.content.length > 0
      ).length,
      averageContentPerSection:
        structure.sections.length > 0
          ? Math.round(
              structure.sections.reduce((sum, s) => sum + s.content.length, 0) /
                structure.sections.length
            )
          : 0,
    };
  }

  /**
   * Clean and normalize extracted text
   * @param {string} text - Raw text to clean
   * @returns {string} - Cleaned text
   */
  _cleanText(text) {
    if (!text || typeof text !== "string") return "";

    return text
      .replace(/\s+/g, " ") // Normalize whitespace
      .replace(/[^\w\s\-.,!?:;()]/g, "") // Remove special characters
      .trim();
  }

  /**
   * Extract text content for specific document sections
   * @param {object} structure - Document structure
   * @param {Array} sectionTypes - Types of sections to extract
   * @returns {string} - Extracted text
   */
  extractSectionText(structure, sectionTypes = ["Text", "List-item"]) {
    let sectionText = "";

    for (const section of structure.sections) {
      sectionText += `\n--- ${section.title} ---\n`;

      for (const content of section.content) {
        if (sectionTypes.includes(content.type)) {
          sectionText += content.text + "\n";
        }
      }
    }

    return sectionText.trim();
  }

  /**
   * Create searchable index of document content
   * @param {object} structure - Document structure
   * @returns {Array} - Array of searchable elements
   */
  createSearchIndex(structure) {
    const searchIndex = [];

    // Add title if exists
    if (structure.title) {
      searchIndex.push({
        type: "title",
        text: structure.title,
        page: 1,
        section: null,
        importance: 1.0,
      });
    }

    // Add sections and content
    for (const section of structure.sections) {
      // Add section header
      searchIndex.push({
        type: "section-header",
        text: section.title,
        page: section.page,
        section: section.index,
        importance: 0.8,
      });

      // Add section content
      for (const content of section.content) {
        searchIndex.push({
          type: content.type.toLowerCase(),
          text: content.text,
          page: content.page,
          section: section.index,
          importance: content.type === "Text" ? 0.6 : 0.4,
        });
      }
    }

    return searchIndex;
  }

  /**
   * Save document structure to JSON file
   * @param {object} structure - Document structure to save
   * @param {string} filename - Original filename
   * @param {string} outputDir - Output directory
   * @returns {Promise<string>} - Path to saved file
   */
  async saveDocumentStructure(
    structure,
    filename,
    outputDir = "./parsed_jsons"
  ) {
    await fs.ensureDir(outputDir);

    const outputPath = path.join(outputDir, `${filename}.json`);
    const documentData = {
      filename: filename,
      title: structure.title || "Untitled Document",
      structure: structure,
      searchIndex: this.createSearchIndex(structure),
      metadata: {
        ...structure.metadata,
        generatedAt: new Date().toISOString(),
        analyzer: "document-analyzer-v1.0",
      },
    };

    await fs.outputJSON(outputPath, documentData, { spaces: 2 });
    console.log(`💾 [Document Analyzer] Structure saved: ${outputPath}`);

    return outputPath;
  }

  /**
   * Analyze document quality and provide insights
   * @param {object} structure - Document structure
   * @param {object} statistics - Document statistics
   * @returns {object} - Quality analysis
   */
  analyzeDocumentQuality(structure, statistics) {
    const quality = {
      overall: "good",
      issues: [],
      suggestions: [],
      scores: {},
    };

    // Structure quality
    if (structure.sections.length === 0) {
      quality.issues.push(
        "No sections detected - document may lack clear structure"
      );
      quality.scores.structure = 0.2;
    } else if (structure.sections.length < 3) {
      quality.suggestions.push(
        "Consider adding more section headers for better organization"
      );
      quality.scores.structure = 0.6;
    } else {
      quality.scores.structure = 0.9;
    }

    // Content quality
    if (statistics.wordCount < 100) {
      quality.issues.push("Very low word count - document may be incomplete");
      quality.scores.content = 0.3;
    } else if (statistics.wordCount < 500) {
      quality.scores.content = 0.6;
    } else {
      quality.scores.content = 0.9;
    }

    // Title quality
    if (!structure.title) {
      quality.issues.push("No document title detected");
      quality.scores.title = 0.0;
    } else if (structure.title.length < 5) {
      quality.suggestions.push("Document title seems very short");
      quality.scores.title = 0.5;
    } else {
      quality.scores.title = 1.0;
    }

    // Calculate overall score
    const scores = Object.values(quality.scores);
    const overallScore =
      scores.reduce((sum, score) => sum + score, 0) / scores.length;

    if (overallScore >= 0.8) quality.overall = "excellent";
    else if (overallScore >= 0.6) quality.overall = "good";
    else if (overallScore >= 0.4) quality.overall = "fair";
    else quality.overall = "poor";

    quality.scores.overall = overallScore;

    return quality;
  }
}

export default DocumentAnalyzer;
