// ollama-handler.js - Ollama API Handler for Text Processing
import { spawn } from "child_process";

export class OllamaHandler {
  constructor(modelName = "gbenson/qwen2.5-0.5b-instruct") {
    this.modelName = modelName;
    this.maxRetries = 3;
    this.timeoutMs = 60000; // 60 seconds
  }

  /**
   * Execute Ollama command with the specified model
   * @param {string} prompt - The prompt to send to the model
   * @param {object} options - Additional options
   * @returns {Promise<string>} - Model response
   */
  async executePrompt(prompt, options = {}) {
    const maxRetries = options.maxRetries || this.maxRetries;
    const timeoutMs = options.timeoutMs || this.timeoutMs;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(
          `🤖 [Ollama] Attempt ${attempt}/${maxRetries} - Using model: ${this.modelName}`
        );

        const response = await this._runOllamaCommand(prompt, timeoutMs);

        if (response && response.trim()) {
          console.log(`✅ [Ollama] Success on attempt ${attempt}`);
          return response.trim();
        } else {
          throw new Error("Empty response from Ollama");
        }
      } catch (error) {
        console.warn(`⚠️ [Ollama] Attempt ${attempt} failed:`, error.message);

        if (attempt === maxRetries) {
          throw new Error(
            `Ollama failed after ${maxRetries} attempts: ${error.message}`
          );
        }

        // Wait before retry
        await this._sleep(1000 * attempt);
      }
    }
  }

  /**
   * Run Ollama command as child process
   * @param {string} prompt - The prompt to send
   * @param {number} timeoutMs - Timeout in milliseconds
   * @returns {Promise<string>} - Model response
   */
  _runOllamaCommand(prompt, timeoutMs) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (ollamaProcess && !ollamaProcess.killed) {
          ollamaProcess.kill("SIGTERM");
        }
        reject(new Error(`Ollama timeout after ${timeoutMs}ms`));
      }, timeoutMs);

      const ollamaProcess = spawn("ollama", ["run", this.modelName], {
        shell: true,
      });

      let output = "";
      let error = "";

      // Send prompt to stdin
      ollamaProcess.stdin.write(prompt);
      ollamaProcess.stdin.end();

      ollamaProcess.stdout.on("data", (data) => {
        output += data.toString();
      });

      ollamaProcess.stderr.on("data", (data) => {
        error += data.toString();
      });

      ollamaProcess.on("close", (code) => {
        clearTimeout(timeout);

        if (code === 0) {
          resolve(output);
        } else {
          reject(
            new Error(`Ollama process exited with code ${code}: ${error}`)
          );
        }
      });

      ollamaProcess.on("error", (err) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to start Ollama process: ${err.message}`));
      });
    });
  }

  /**
   * Summarize document text
   * @param {string} text - Text content to summarize
   * @param {string} pdfName - Name of the PDF for context
   * @returns {Promise<object>} - Summary object with pdf_name and pdf_summary
   */
  async summarizeDocument(text, pdfName) {
    if (!text || text.trim().length < 50) {
      return {
        pdf_name: pdfName,
        pdf_summary: `Brief document: ${pdfName} - insufficient content for meaningful summarization`,
      };
    }

    const prompt = `Please provide a concise summary of the following document content. Focus on the main topics, key information, and important sections.

Document: ${pdfName}
Content:
${text.substring(0, 4000)} ${text.length > 4000 ? "...(truncated)" : ""}

Instructions:
- Provide a clear, informative summary in 2-4 sentences
- Focus on actionable content and main themes
- Avoid repetitive information
- Keep the summary professional and concise

Summary:`;

    try {
      const summary = await this.executePrompt(prompt, { timeoutMs: 45000 });

      return {
        pdf_name: pdfName,
        pdf_summary: summary,
      };
    } catch (error) {
      console.warn(
        `⚠️ [Ollama] Summarization failed for ${pdfName}:`,
        error.message
      );

      // Fallback summary
      const words = text.split(" ").slice(0, 50).join(" ");
      return {
        pdf_name: pdfName,
        pdf_summary: `Document ${pdfName}: ${words}...`,
      };
    }
  }

  /**
   * Generate ranked topics based on summaries, persona, and task
   * @param {Array} summaries - Array of summary objects
   * @param {string} persona - User persona/role
   * @param {string} task - Task to be accomplished
   * @returns {Promise<Array>} - Array of ranked topics
   */
  async generateRankedTopics(summaries, persona, task) {
    const summariesText = summaries
      .map((s, i) => `${i + 1}. ${s.pdf_name}: ${s.pdf_summary}`)
      .join("\n");

    const prompt = `You are helping a ${persona} with the following task: ${task}

Based on the following document summaries, identify and rank the 5-7 most important section topics that would be most relevant for this persona and task.

Document Summaries:
${summariesText}

Instructions:
- Consider what would be most actionable and relevant for a ${persona}
- Focus on topics that directly support the task: ${task}
- Rank topics by importance (most important first)
- Each topic should be a concise phrase or title (2-8 words)
- Provide ONLY the topics as a simple list, one per line
- Do not include numbers, bullets, or explanations

Example format:
Introduction and Overview
Main Content and Procedures
Requirements and Guidelines
Implementation Steps
Best Practices and Standards

Topics:`;

    try {
      const response = await this.executePrompt(prompt, { timeoutMs: 60000 });

      // Parse the response into an array of topics
      const topics = response
        .split("\n")
        .map((line) => line.trim())
        .filter(
          (line) => line.length > 0 && !line.toLowerCase().includes("topics:")
        )
        .slice(0, 7); // Limit to 7 topics max

      if (topics.length > 0) {
        console.log(`✅ [Ollama] Generated ${topics.length} topics:`, topics);
        return topics;
      } else {
        throw new Error("No valid topics generated");
      }
    } catch (error) {
      console.warn(`⚠️ [Ollama] Topic generation failed:`, error.message);

      // Fallback topics
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
   * Enhance semantic matching with AI-powered text analysis
   * @param {string} topic - Topic to match
   * @param {string} text - Text content to analyze
   * @returns {Promise<number>} - Relevance score (0-1)
   */
  async getSemanticRelevance(topic, text) {
    if (!text || text.length < 10) {
      return 0;
    }

    const prompt = `Analyze how relevant the following text content is to the given topic. Rate the relevance on a scale from 0.0 to 1.0.

Topic: "${topic}"

Text Content:
${text.substring(0, 500)}${text.length > 500 ? "..." : ""}

Instructions:
- Consider semantic meaning, not just keyword matching
- 0.0 = Not relevant at all
- 0.5 = Somewhat relevant
- 1.0 = Very relevant and directly related
- Provide ONLY the numeric score (e.g., 0.75)

Relevance Score:`;

    try {
      const response = await this.executePrompt(prompt, { timeoutMs: 30000 });
      const score = parseFloat(response.match(/\d*\.?\d+/)?.[0] || "0");
      return Math.max(0, Math.min(1, score)); // Ensure score is between 0 and 1
    } catch (error) {
      console.warn(
        `⚠️ [Ollama] Semantic relevance calculation failed:`,
        error.message
      );
      return 0;
    }
  }

  /**
   * Sleep utility function
   * @param {number} ms - Milliseconds to sleep
   * @returns {Promise} - Sleep promise
   */
  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Test Ollama connection and model availability
   * @returns {Promise<boolean>} - True if Ollama is available
   */
  async testConnection() {
    try {
      console.log("🔄 [Ollama] Testing connection...");
      const testResponse = await this.executePrompt(
        'Say "Hello" if you can hear me.',
        {
          maxRetries: 1,
          timeoutMs: 15000,
        }
      );

      const isWorking = testResponse.toLowerCase().includes("hello");

      if (isWorking) {
        console.log("✅ [Ollama] Connection test successful");
      } else {
        console.warn(
          "⚠️ [Ollama] Connection test failed - unexpected response"
        );
      }

      return isWorking;
    } catch (error) {
      console.warn("⚠️ [Ollama] Connection test failed:", error.message);
      return false;
    }
  }
}

export default OllamaHandler;
