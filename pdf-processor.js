// pdf-processor.js - PDF Processing with YOLO and OCR
import fs from "fs-extra";
import path from "path";
import { createCanvas, loadImage } from "canvas";
import { AutoModel, AutoProcessor, RawImage } from "@huggingface/transformers";

export class PDFProcessor {
  constructor(config = {}) {
    this.tempDir = config.tempDir || "./temp_images";
    this.confidenceThreshold = config.confidenceThreshold || 0.5;
    this.ocrConfig = {
      enabled: false,
      language: "eng",
    };

    // YOLO model components
    this.model = null;
    this.processor = null;
    this.ocrWorker = null;

    // Document layout labels
    this.id2label = {
      0: "Caption",
      1: "Footnote",
      2: "Formula",
      3: "List-item",
      4: "Page-footer",
      5: "Page-header",
      6: "Picture",
      7: "Section-header",
      8: "Table",
      9: "Text",
      10: "Title",
    };

    this.isInitialized = false;
  }

  /**
   * Initialize YOLO model and OCR components
   */
  async initialize() {
    if (this.isInitialized) {
      return;
    }

    console.log("🔄 [PDF Processor] Initializing YOLO model and OCR...");

    try {
      // Load YOLO model for document layout analysis
      this.model = await AutoModel.from_pretrained(
        "Xenova/yolov10m-doclaynet",
        {
          quantized: false,
        }
      );
      this.processor = await AutoProcessor.from_pretrained(
        "Xenova/yolov10m-doclaynet"
      );
      console.log("✅ [PDF Processor] YOLO model loaded successfully");
    } catch (error) {
      console.error(
        "❌ [PDF Processor] Failed to load YOLO model:",
        error.message
      );
      throw error;
    }

    // Try to initialize OCR
    await this._initializeOCR();

    // Ensure temp directory exists
    await fs.ensureDir(this.tempDir);

    this.isInitialized = true;
    console.log("✅ [PDF Processor] Initialization complete");
  }

  /**
   * Initialize OCR worker if available
   */
  async _initializeOCR() {
    try {
      const tesseractModule = await import("tesseract.js");
      const { createWorker } = tesseractModule;

      this.ocrWorker = await createWorker(this.ocrConfig.language);
      this.ocrConfig.enabled = true;
      console.log("✅ [PDF Processor] OCR worker initialized");
    } catch (error) {
      console.warn("⚠️ [PDF Processor] OCR not available:", error.message);
      this.ocrConfig.enabled = false;
    }
  }

  /**
   * Convert PDF to images using pdf-poppler
   * @param {string} pdfPath - Path to PDF file
   * @param {string} outPrefix - Output file prefix
   * @returns {Promise<Array>} - Array of image file names
   */
  async convertPdfToImages(pdfPath, outPrefix) {
    try {
      const pdfPopplerModule = await import("pdf-poppler");
      const pdfPoppler = pdfPopplerModule.default;

      const options = {
        format: "png",
        out_dir: this.tempDir,
        out_prefix: outPrefix,
        page: null,
      };

      await fs.ensureDir(this.tempDir);
      await pdfPoppler.convert(pdfPath, options);

      const imageFiles = (await fs.readdir(this.tempDir))
        .filter((f) => f.startsWith(outPrefix) && f.endsWith(".png"))
        .sort();

      console.log(
        `✅ [PDF Processor] Converted ${pdfPath} to ${imageFiles.length} images`
      );
      return imageFiles;
    } catch (error) {
      console.error(
        `❌ [PDF Processor] PDF conversion failed for ${pdfPath}:`,
        error.message
      );
      throw error;
    }
  }

  /**
   * Extract text from image region using OCR
   * @param {string} imagePath - Path to image file
   * @returns {Promise<string>} - Extracted text
   */
  async extractTextFromImage(imagePath) {
    if (!this.ocrConfig.enabled || !this.ocrWorker) {
      return "OCR not available";
    }

    try {
      const {
        data: { text },
      } = await this.ocrWorker.recognize(imagePath);
      return text.trim();
    } catch (error) {
      console.warn(
        `⚠️ [PDF Processor] OCR extraction failed for ${imagePath}:`,
        error.message
      );
      return "OCR extraction failed";
    }
  }

  /**
   * Process single page with YOLO object detection and OCR
   * @param {string} imagePath - Path to page image
   * @param {number} pageNumber - Page number
   * @returns {Promise<object>} - Page processing result
   */
  async processPage(imagePath, pageNumber) {
    if (!this.isInitialized) {
      throw new Error(
        "PDF Processor not initialized. Call initialize() first."
      );
    }

    const startTime = performance.now();

    try {
      console.log(`🔄 [PDF Processor] Processing page ${pageNumber}...`);

      // Load and process image with YOLO
      const image = await RawImage.read(imagePath);
      const { pixel_values } = await this.processor(image);
      const { outputs } = await this.model({ images: pixel_values });

      const sizes = [[image.height, image.width]];
      const { boxes, scores, labels } =
        await this.processor.post_process_object_detection(
          outputs,
          this.confidenceThreshold,
          sizes
        );

      const detections = [];

      // Process each detection
      for (let i = 0; i < boxes.data.length / 4; i++) {
        const bbox = [
          boxes.data[i * 4],
          boxes.data[i * 4 + 1],
          boxes.data[i * 4 + 2],
          boxes.data[i * 4 + 3],
        ];
        const score = scores.data[i];
        const label = this.id2label[labels.data[i]];

        // Extract text from detected region if OCR is available
        let extractedText = "";
        if (
          this.ocrConfig.enabled &&
          ["Text", "Title", "Section-header", "List-item"].includes(label)
        ) {
          try {
            // Crop the detected region for OCR
            const canvas = createCanvas(bbox[2] - bbox[0], bbox[3] - bbox[1]);
            const ctx = canvas.getContext("2d");
            const fullImage = await loadImage(imagePath);

            ctx.drawImage(
              fullImage,
              bbox[0],
              bbox[1],
              bbox[2] - bbox[0],
              bbox[3] - bbox[1],
              0,
              0,
              bbox[2] - bbox[0],
              bbox[3] - bbox[1]
            );

            const croppedBuffer = canvas.toBuffer();
            const tempCropPath = path.join(
              this.tempDir,
              `crop_${pageNumber}_${i}.png`
            );
            await fs.writeFile(tempCropPath, croppedBuffer);

            extractedText = await this.extractTextFromImage(tempCropPath);

            // Clean up temp crop file
            await fs.unlink(tempCropPath);
          } catch (error) {
            console.warn(
              `⚠️ [PDF Processor] Text extraction failed for detection ${i}:`,
              error.message
            );
            extractedText = "Text extraction failed";
          }
        }

        const detection = {
          bbox: bbox,
          normalizedBbox: [
            parseFloat((bbox[0] / image.width).toFixed(4)),
            parseFloat((bbox[1] / image.height).toFixed(4)),
            parseFloat((bbox[2] / image.width).toFixed(4)),
            parseFloat((bbox[3] / image.height).toFixed(4)),
          ],
          label: label,
          confidence: parseFloat(score.toFixed(3)),
          area: Math.round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
          center: [
            Math.round((bbox[0] + bbox[2]) / 2),
            Math.round((bbox[1] + bbox[3]) / 2),
          ],
          width: Math.round(bbox[2] - bbox[0]),
          height: Math.round(bbox[3] - bbox[1]),
          extractedText: extractedText,
          pageNumber: pageNumber,
        };

        detections.push(detection);
      }

      // Sort detections by reading order (top to bottom, left to right)
      detections.sort((a, b) => {
        const yDiff = a.center[1] - b.center[1];
        if (Math.abs(yDiff) > 20) return yDiff;
        return a.center[0] - b.center[0];
      });

      // Add reading order index
      detections.forEach((detection, index) => {
        detection.reading_order = index + 1;
      });

      const processingTime = ((performance.now() - startTime) / 1000).toFixed(
        2
      );
      console.log(
        `✅ [PDF Processor] Page ${pageNumber} processed in ${processingTime}s - Found ${detections.length} elements`
      );

      return {
        pageNumber,
        processingTime: parseFloat(processingTime),
        detections: detections,
        sourceImagePath: imagePath,
        imageWidth: image.width,
        imageHeight: image.height,
      };
    } catch (error) {
      console.error(
        `❌ [PDF Processor] Error processing page ${pageNumber}:`,
        error.message
      );
      throw error;
    }
  }

  /**
   * Process entire PDF document
   * @param {object} doc - Document object with filename and title
   * @returns {Promise<object>} - Processing result with structure and metadata
   */
  async processPDF(doc) {
    console.log(`🔄 [PDF Processor] Processing PDF: ${doc.filename}`);

    const filename = doc.filename;
    const outPrefix = path.parse(filename).name;

    try {
      // Convert PDF to images
      const imageFiles = await this.convertPdfToImages(
        `./${filename}`,
        outPrefix
      );

      // Process all pages in parallel for better performance
      const pageProcessingPromises = imageFiles.map((imageFile, index) => {
        const imagePath = path.join(this.tempDir, imageFile);
        const pageNumber = index + 1;
        return this.processPage(imagePath, pageNumber);
      });

      const pageResults = await Promise.all(pageProcessingPromises);

      console.log(
        `✅ [PDF Processor] Completed processing ${filename} - ${pageResults.length} pages`
      );

      return {
        filename,
        title: doc.title,
        pageResults,
        metadata: {
          totalPages: pageResults.length,
          totalElements: pageResults.reduce(
            (sum, page) => sum + page.detections.length,
            0
          ),
          totalProcessingTime: pageResults.reduce(
            (sum, page) => sum + page.processingTime,
            0
          ),
        },
      };
    } catch (error) {
      console.error(
        `❌ [PDF Processor] Failed to process ${filename}:`,
        error.message
      );
      throw error;
    }
  }

  /**
   * Process multiple PDFs in parallel
   * @param {Array} documents - Array of document objects
   * @returns {Promise<Array>} - Array of processing results
   */
  async processMultiplePDFs(documents) {
    console.log(
      `🔄 [PDF Processor] Processing ${documents.length} PDFs in parallel...`
    );

    const processingPromises = documents.map((doc) => this.processPDF(doc));
    const results = await Promise.all(processingPromises);

    console.log(
      `✅ [PDF Processor] Completed processing all ${documents.length} PDFs`
    );
    return results;
  }

  /**
   * Clean up resources and temporary files
   */
  async cleanup() {
    try {
      // Terminate OCR worker
      if (this.ocrWorker) {
        await this.ocrWorker.terminate();
        this.ocrWorker = null;
        console.log("🧹 [PDF Processor] OCR worker terminated");
      }

      // Clean up temp directory
      await fs.rm(this.tempDir, { recursive: true, force: true });
      console.log("🧹 [PDF Processor] Temporary files cleaned up");
    } catch (error) {
      console.warn("⚠️ [PDF Processor] Cleanup failed:", error.message);
    }
  }

  /**
   * Get processor statistics
   * @returns {object} - Processing statistics
   */
  getStats() {
    return {
      initialized: this.isInitialized,
      ocrEnabled: this.ocrConfig.enabled,
      confidenceThreshold: this.confidenceThreshold,
      tempDir: this.tempDir,
    };
  }
}

export default PDFProcessor;
