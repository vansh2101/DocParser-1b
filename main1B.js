// main1B.js - Modular Round 1B Pipeline with Ollama Integration
import fs from "fs-extra";
import path from "path";

// Import modular components
import OllamaHandler from "./ollama-handler.js";
import PDFProcessor from "./pdf-processor.js";
import DocumentAnalyzer from "./document-analyzer.js";
import SemanticMatcher from "./semantic-matcher.js";

// Configuration
const CONFIG = {
  input: "./challenge1b_input.json",
  output: {
    directory: "./output",
    finalOutput: "./output.json",
    summaries: "./summaries.json",
    parsedJsons: "./parsed_jsons",
  },
  processing: {
    tempDir: "./temp_images",
    confidenceThreshold: 0.5,
    minTextLength: 10,
  },
  ollama: {
    modelName: "gbenson/qwen2.5-0.5b-instruct",
    maxRetries: 3,
    timeoutMs: 60000,
  },
  matching: {
    minMatchScore: 0.1,
    aiEnhancedMode: true,
    fuzzyWeight: 0.3,
    cosineWeight: 0.4,
    aiWeight: 0.3,
    maxConcurrentAIRequests: 3,
  },
};

export class Round1BPipeline {
  constructor(config = CONFIG) {
    this.config = { ...CONFIG, ...config };

    // Initialize components
    this.ollamaHandler = new OllamaHandler(this.config.ollama.modelName);
    this.pdfProcessor = new PDFProcessor({
      tempDir: this.config.processing.tempDir,
      confidenceThreshold: this.config.processing.confidenceThreshold,
    });
    this.documentAnalyzer = new DocumentAnalyzer({
      minTextLength: this.config.processing.minTextLength,
    });
    this.semanticMatcher = new SemanticMatcher(
      this.ollamaHandler,
      this.config.matching
    );

    // Pipeline state
    this.stats = {
      totalDocuments: 0,
      processedDocuments: 0,
      totalMatches: 0,
      startTime: null,
      endTime: null,
    };
  }

  /**
   * Run the complete Round 1B pipeline
   */
  async run() {
    const startTime = performance.now();
    this.stats.startTime = new Date().toISOString();

    try {
      console.log("🚀 Starting Modular Round 1B Pipeline with Ollama");
      console.log("=".repeat(60));

      // Step 1: Initialize all components
      await this.initializeComponents();

      // Step 2: Load and validate input
      const input = await this.loadInput();
      this.stats.totalDocuments = input.documents.length;

      console.log(`📄 Processing ${input.documents.length} document(s)`);
      console.log(`👤 Persona: ${input.persona.role}`);
      console.log(`🎯 Task: ${input.job_to_be_done.task}`);

      // Step 3: Process PDFs in parallel
      console.log("\n🔄 Step 1: Processing PDFs with YOLO + OCR...");
      const pdfResults = await this.processPDFs(input.documents);

      // Step 4: Analyze document structures and extract text
      console.log("\n📊 Step 2: Analyzing document structures...");
      const analysisResults = await this.analyzeDocuments(pdfResults);

      // Step 5: Generate summaries with Ollama
      console.log("\n📝 Step 3: Generating summaries with Ollama...");
      const summaries = await this.generateSummaries(analysisResults);

      // Step 6: Generate ranked topics with Ollama
      console.log("\n🤖 Step 4: Generating ranked topics with Ollama...");
      const rankedTopics = await this.generateRankedTopics(
        summaries,
        input.persona.role,
        input.job_to_be_done.task
      );

      // Step 7: Perform semantic matching
      console.log("\n🔍 Step 5: Performing AI-enhanced semantic matching...");
      const matches = await this.performSemanticMatching(
        rankedTopics,
        analysisResults
      );

      // Step 8: Generate final output
      console.log("\n📄 Step 6: Generating final output...");
      const output = await this.generateFinalOutput(
        input,
        summaries,
        rankedTopics,
        matches,
        startTime
      );

      // Step 9: Save all outputs
      await this.saveAllOutputs(output, summaries, analysisResults);

      // Step 10: Display results
      this.displayResults(output, performance.now() - startTime);

      return output;
    } catch (error) {
      console.error("❌ Pipeline failed:", error.message);
      console.error(error.stack);
      throw error;
    } finally {
      // Always cleanup resources
      await this.cleanup();
      this.stats.endTime = new Date().toISOString();
    }
  }

  /**
   * Initialize all pipeline components
   */
  async initializeComponents() {
    console.log("🔄 Initializing pipeline components...");

    // Test Ollama connection
    const ollamaAvailable = await this.ollamaHandler.testConnection();
    if (!ollamaAvailable) {
      console.warn(
        "⚠️ Ollama not available - falling back to basic processing"
      );
      this.config.matching.aiEnhancedMode = false;
    }

    // Initialize PDF processor
    await this.pdfProcessor.initialize();

    // Ensure output directories exist
    await fs.ensureDir(this.config.output.directory);
    await fs.ensureDir(this.config.output.parsedJsons);

    console.log("✅ All components initialized successfully");
  }

  /**
   * Load and validate input configuration
   */
  async loadInput() {
    try {
      const input = await fs.readJSON(this.config.input);

      // Validate input structure
      if (!input.documents || !Array.isArray(input.documents)) {
        throw new Error("Invalid input: documents array is required");
      }
      if (!input.persona || !input.persona.role) {
        throw new Error("Invalid input: persona.role is required");
      }
      if (!input.job_to_be_done || !input.job_to_be_done.task) {
        throw new Error("Invalid input: job_to_be_done.task is required");
      }

      return input;
    } catch (error) {
      throw new Error(
        `Failed to load input from ${this.config.input}: ${error.message}`
      );
    }
  }

  /**
   * Process all PDFs with YOLO and OCR
   */
  async processPDFs(documents) {
    const results = await this.pdfProcessor.processMultiplePDFs(documents);
    this.stats.processedDocuments = results.length;

    console.log(`✅ Processed ${results.length} PDFs successfully`);
    return results;
  }

  /**
   * Analyze document structures and extract text
   */
  async analyzeDocuments(pdfResults) {
    const analysisResults = [];

    for (const pdfResult of pdfResults) {
      console.log(`📊 Analyzing structure for ${pdfResult.filename}...`);

      const analysis = this.documentAnalyzer.createDocumentStructure(
        pdfResult.pageResults
      );

      // Save document structure
      await this.documentAnalyzer.saveDocumentStructure(
        analysis.structure,
        pdfResult.filename,
        this.config.output.parsedJsons
      );

      analysisResults.push({
        filename: pdfResult.filename,
        title: pdfResult.title,
        structure: analysis.structure,
        allText: analysis.allText,
        statistics: analysis.statistics,
        quality: this.documentAnalyzer.analyzeDocumentQuality(
          analysis.structure,
          analysis.statistics
        ),
      });

      console.log(
        `✅ Analysis complete for ${pdfResult.filename}: ${analysis.statistics.wordCount} words, ${analysis.statistics.totalSections} sections`
      );
    }

    return analysisResults;
  }

  /**
   * Generate summaries using Ollama
   */
  async generateSummaries(analysisResults) {
    const summaryPromises = analysisResults.map(async (analysis) => {
      try {
        console.log(`📝 Summarizing ${analysis.filename}...`);
        return await this.ollamaHandler.summarizeDocument(
          analysis.allText,
          analysis.filename
        );
      } catch (error) {
        console.warn(
          `⚠️ Summary failed for ${analysis.filename}:`,
          error.message
        );
        return {
          pdf_name: analysis.filename,
          pdf_summary: `Document analysis for ${analysis.filename} - ${analysis.statistics.wordCount} words across ${analysis.statistics.totalSections} sections`,
        };
      }
    });

    const summaries = await Promise.all(summaryPromises);
    console.log(`✅ Generated ${summaries.length} summaries`);

    return summaries;
  }

  /**
   * Generate ranked topics using Ollama
   */
  async generateRankedTopics(summaries, persona, task) {
    try {
      const topics = await this.ollamaHandler.generateRankedTopics(
        summaries,
        persona,
        task
      );
      console.log(`✅ Generated ${topics.length} ranked topics:`, topics);
      return topics;
    } catch (error) {
      console.warn(
        "⚠️ Topic generation failed, using fallback topics:",
        error.message
      );
      return [
        "Introduction and Overview",
        "Main Content and Procedures",
        "Requirements and Guidelines",
        "Implementation Steps",
        "Best Practices and Standards",
      ];
    }
  }

  /**
   * Perform semantic matching between topics and document content
   */
  async performSemanticMatching(topics, analysisResults) {
    const allMatches = [];

    for (const analysis of analysisResults) {
      console.log(`🔍 Matching topics for ${analysis.filename}...`);

      const matches = await this.semanticMatcher.findBestMatches(
        topics,
        analysis.structure,
        analysis.filename
      );

      allMatches.push(...matches);
    }

    // Sort all matches by importance and score
    allMatches.sort((a, b) => {
      if (a.importance_rank !== b.importance_rank) {
        return a.importance_rank - b.importance_rank;
      }
      return b.match_score - a.match_score;
    });

    this.stats.totalMatches = allMatches.length;
    console.log(
      `✅ Found ${allMatches.length} total matches across all documents`
    );

    return allMatches;
  }

  /**
   * Generate final output structure
   */
  async generateFinalOutput(
    input,
    summaries,
    rankedTopics,
    matches,
    startTime
  ) {
    const processingTime = ((performance.now() - startTime) / 1000).toFixed(2);
    const refinedAnalysis =
      this.semanticMatcher.generateRefinedAnalysis(matches);

    return {
      metadata: {
        input_documents: input.documents.map((d) => d.filename),
        persona: input.persona.role,
        job_to_be_done: input.job_to_be_done.task,
        processing_timestamp: new Date().toISOString(),
        total_matches: matches.length,
        ranked_topics: rankedTopics,
        processing_time_seconds: parseFloat(processingTime),
        pipeline_version: "modular-v2.0",
        ai_model: this.config.ollama.modelName,
        statistics: {
          documents_processed: this.stats.processedDocuments,
          semantic_matcher_stats: this.semanticMatcher.getStats(),
          pdf_processor_stats: this.pdfProcessor.getStats(),
        },
      },
      extracted_sections: matches,
      subsection_analysis: refinedAnalysis,
    };
  }

  /**
   * Save all output files
   */
  async saveAllOutputs(output, summaries, analysisResults) {
    // Save final output
    await fs.outputJSON(this.config.output.finalOutput, output, { spaces: 2 });
    console.log(`💾 Final output saved: ${this.config.output.finalOutput}`);

    // Save summaries
    await fs.outputJSON(this.config.output.summaries, summaries, { spaces: 2 });
    console.log(`💾 Summaries saved: ${this.config.output.summaries}`);

    // Document structures are already saved by DocumentAnalyzer
    console.log(
      `💾 Document structures saved in: ${this.config.output.parsedJsons}/`
    );
  }

  /**
   * Display final results and statistics
   */
  displayResults(output, totalTime) {
    const timeSeconds = (totalTime / 1000).toFixed(2);

    console.log("\n" + "=".repeat(60));
    console.log("🎉 MODULAR PIPELINE COMPLETE");
    console.log("=".repeat(60));
    console.log(`⏱️  Total Processing Time: ${timeSeconds}s`);
    console.log(`📊 Documents Processed: ${this.stats.processedDocuments}`);
    console.log(`🔍 Total Matches Found: ${this.stats.totalMatches}`);
    console.log(`🤖 AI Model Used: ${this.config.ollama.modelName}`);
    console.log(`📄 Final Output: ${this.config.output.finalOutput}`);
    console.log(`📝 Summaries: ${this.config.output.summaries}`);
    console.log(`📁 Parsed Structures: ${this.config.output.parsedJsons}/`);

    console.log("\n🏆 Top 5 Matches:");
    output.extracted_sections.slice(0, 5).forEach((match, index) => {
      console.log(
        `  ${index + 1}. ${match.document} (Page ${match.page_number})`
      );
      console.log(
        `     Topic: "${match.topic}" (Rank ${match.importance_rank})`
      );
      console.log(
        `     Score: ${match.match_score} | Type: ${match.match_type}`
      );
      console.log(`     AI Enhanced: ${match.ai_score > 0 ? "Yes" : "No"}`);
      console.log(`     Section: ${match.section_title.substring(0, 60)}...`);
    });

    console.log("\n📈 Component Statistics:");
    const matcherStats = this.semanticMatcher.getStats();
    console.log(`  Semantic Matcher:`);
    console.log(`    - Total matches: ${matcherStats.totalMatches}`);
    console.log(`    - AI enhanced: ${matcherStats.aiEnhancedMatches}`);
    console.log(`    - Traditional only: ${matcherStats.traditionalMatches}`);
    if (matcherStats.failedAIMatches > 0) {
      console.log(`    - Failed AI requests: ${matcherStats.failedAIMatches}`);
    }

    const processorStats = this.pdfProcessor.getStats();
    console.log(`  PDF Processor:`);
    console.log(`    - OCR enabled: ${processorStats.ocrEnabled}`);
    console.log(
      `    - Confidence threshold: ${processorStats.confidenceThreshold}`
    );
  }

  /**
   * Clean up resources and temporary files
   */
  async cleanup() {
    try {
      await this.pdfProcessor.cleanup();
      console.log("🧹 Pipeline cleanup completed");
    } catch (error) {
      console.warn("⚠️ Cleanup failed:", error.message);
    }
  }

  /**
   * Get pipeline statistics
   */
  getStats() {
    return {
      ...this.stats,
      components: {
        semanticMatcher: this.semanticMatcher.getStats(),
        pdfProcessor: this.pdfProcessor.getStats(),
      },
    };
  }
}

/**
 * Main execution function
 */
async function main() {
  try {
    // Create and run pipeline
    const pipeline = new Round1BPipeline();
    const result = await pipeline.run();

    console.log("\n✅ Pipeline execution completed successfully!");
    process.exit(0);
  } catch (error) {
    console.error("\n❌ Pipeline execution failed:");
    console.error(error.message);
    process.exit(1);
  }
}

// Run the pipeline if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export default Round1BPipeline;
