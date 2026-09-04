"use client";

import { useState } from "react";

interface Detection {
  class: number;
  label: string;
  confidence: number;
  fuzzy_score: number;
  size_ratio?: number;
  bbox: number[];
}

interface DetectionResponse {
  success: boolean;
  metrics: {
    yolo: number;
    fuzzy: number;
    hybrid: number;
  };
  best_threshold: number;
  ga_history: number[];
  detections: Detection[];
  image: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image.");
      return;
    }

    setFile(selectedFile);
    setResult(null);
    setError(null);

    const url = URL.createObjectURL(selectedFile);
    setPreview(url);
  };

  const handleDetect = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const apiUrl =
        process.env.NEXT_PUBLIC_API_URL ||
        "http://localhost:8000";

      const response = await fetch(`${apiUrl}/detect`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Detection failed.");
      }

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the detection server."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white">

      {/* HEADER */}
      <header className="border-b border-slate-800">
        <div className="mx-auto max-w-7xl px-6 py-12 text-center">

          <div className="mb-4 text-5xl">
            🚀
          </div>

          <h1 className="text-4xl font-bold md:text-6xl">
            Hybrid Object Detection
          </h1>

          <p className="mt-4 text-lg text-slate-400">
            YOLO + Fuzzy Logic + Genetic Algorithm
          </p>

          <p className="mt-2 text-sm text-slate-500">
            Intelligent multi-stage object detection system
          </p>

        </div>
      </header>


      <div className="mx-auto max-w-7xl px-6 py-10">

        {/* ==================================================
            UPLOAD SECTION
        ================================================== */}

        <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-2xl">

          <div className="text-center">

            <h2 className="text-3xl font-bold">
              Upload an Image
            </h2>

            <p className="mt-2 text-slate-400">
              Select an image to detect objects
            </p>

          </div>


          {/* FILE UPLOAD */}

          <label className="mx-auto mt-8 flex max-w-3xl cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-950 p-12 transition hover:border-green-500">

            <div className="text-6xl">
              📤
            </div>

            <p className="mt-5 text-lg font-semibold">
              Click to choose an image
            </p>

            <p className="mt-2 text-sm text-slate-500">
              JPG, JPEG or PNG
            </p>

            {file && (
              <p className="mt-4 text-sm text-green-400">
                Selected: {file.name}
              </p>
            )}

            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              className="hidden"
              onChange={(e) => {
                const selected = e.target.files?.[0];

                if (selected) {
                  handleFile(selected);
                }
              }}
            />

          </label>


          {/* IMAGE PREVIEW */}

          {preview && (
            <div className="mx-auto mt-8 max-w-3xl">

              <h3 className="mb-3 text-lg font-semibold">
                📷 Selected Image
              </h3>

              <div className="overflow-hidden rounded-2xl border border-slate-700">

                <img
                  src={preview}
                  alt="Selected image"
                  className="max-h-[550px] w-full object-contain"
                />

              </div>

            </div>
          )}


          {/* DETECT BUTTON */}

          <div className="mt-8 text-center">

            <button
              onClick={handleDetect}
              disabled={!file || loading}
              className="rounded-xl bg-green-500 px-10 py-4 text-lg font-bold text-slate-950 transition hover:bg-green-400 disabled:cursor-not-allowed disabled:opacity-40"
            >

              {loading
                ? "🔄 Processing..."
                : "🔍 Detect Objects"}

            </button>

          </div>


          {/* PROCESSING INFORMATION */}

          <div className="mx-auto mt-6 max-w-3xl rounded-2xl border border-yellow-500/30 bg-yellow-500/5 p-5">

            <h3 className="text-lg font-semibold text-yellow-400">
              ⏳ Why does detection take time?
            </h3>

            <p className="mt-3 text-sm leading-6 text-slate-400">
              Our system uses YOLOv8 with PyTorch for object detection,
              followed by Fuzzy Logic and a Genetic Algorithm for hybrid
              processing. Since the application is running on a free
              CPU-based server, YOLOv8 inference can take longer than usual.
            </p>

            <p className="mt-3 text-sm font-semibold text-slate-300">
              Typical processing time: up to ~1 minute.
              Please keep this page open while your image is being processed.
            </p>

          </div>


          {/* ERROR */}

          {error && (
            <div className="mx-auto mt-6 max-w-3xl rounded-xl border border-red-800 bg-red-950 p-5 text-center text-red-300">

              ❌ {error}

            </div>
          )}

        </section>


        {/* ==================================================
            RESULTS
        ================================================== */}

        {result && (

          <section className="mt-10">

            {/* ==================================================
                ORIGINAL + RESULT IMAGES
            ================================================== */}

            <div className="grid gap-6 md:grid-cols-2">

              {/* ORIGINAL */}

              <div className="rounded-3xl border border-slate-800 bg-slate-900 p-6">

                <h2 className="mb-5 text-2xl font-bold">
                  📷 Original Image
                </h2>

                {preview && (
                  <img
                    src={preview}
                    alt="Original"
                    className="w-full rounded-2xl"
                  />
                )}

              </div>


              {/* PROCESSED */}

              <div className="rounded-3xl border border-slate-800 bg-slate-900 p-6">

                <h2 className="mb-5 text-2xl font-bold">
                  🎯 Detection Result
                </h2>

                <img
                  src={result.image}
                  alt="Detection result"
                  className="w-full rounded-2xl"
                />


                {/* DOWNLOAD */}

                <a
                  href={result.image}
                  download="hybrid-detection-result.jpg"
                  className="mt-5 inline-flex rounded-xl bg-green-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-green-400"
                >
                  ⬇️ Download Result
                </a>

              </div>

            </div>


            {/* ==================================================
                METRICS
            ================================================== */}

            <div className="mt-10">

              <h2 className="mb-6 text-3xl font-bold">
                📊 Performance Metrics
              </h2>

              <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">

                <Metric
                  title="YOLO Detections"
                  value={result.metrics.yolo}
                />

                <Metric
                  title="Fuzzy Detections"
                  value={result.metrics.fuzzy}
                />

                <Metric
                  title="Hybrid Detections"
                  value={result.metrics.hybrid}
                />

                <Metric
                  title="GA Threshold"
                  value={result.best_threshold.toFixed(4)}
                />

              </div>

            </div>


            {/* ==================================================
                GENETIC ALGORITHM
            ================================================== */}

            <div className="mt-8 rounded-3xl border border-slate-800 bg-slate-900 p-6">

              <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">

                <div>

                  <h2 className="text-2xl font-bold">
                    🧬 Genetic Algorithm Optimization
                  </h2>

                  <p className="mt-2 text-slate-400">
                    The genetic algorithm optimized the detection threshold.
                  </p>

                </div>


                <div className="rounded-xl bg-green-500/10 px-5 py-3 text-center">

                  <p className="text-xs text-slate-400">
                    Best Threshold
                  </p>

                  <p className="text-2xl font-bold text-green-400">
                    {result.best_threshold.toFixed(4)}
                  </p>

                </div>

              </div>


              {result.ga_history.length > 0 && (

                <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">

                  {result.ga_history.map((value, index) => (

                    <div
                      key={index}
                      className="rounded-xl border border-slate-700 bg-slate-950 p-4"
                    >

                      <p className="text-sm text-slate-500">
                        Generation {index + 1}
                      </p>

                      <p className="mt-2 text-xl font-bold text-green-400">
                        {value.toFixed(4)}
                      </p>

                    </div>

                  ))}

                </div>

              )}

            </div>


            {/* ==================================================
                DETECTION TABLE
            ================================================== */}

            <div className="mt-8 overflow-hidden rounded-3xl border border-slate-800 bg-slate-900">

              <div className="border-b border-slate-800 p-6">

                <h2 className="text-3xl font-bold">
                  🔎 Detected Objects
                </h2>

                <p className="mt-2 text-slate-400">
                  Objects that passed the hybrid detection stage
                </p>

              </div>


              {result.detections.length === 0 ? (

                <div className="p-8 text-center text-slate-400">
                  No objects passed the hybrid filtering stage.
                </div>

              ) : (

                <div className="overflow-x-auto">

                  <table className="w-full text-left">

                    <thead className="bg-slate-950">

                      <tr className="text-sm text-slate-400">

                        <th className="px-6 py-4">
                          #
                        </th>

                        <th className="px-6 py-4">
                          Object
                        </th>

                        <th className="px-6 py-4">
                          Confidence
                        </th>

                        <th className="px-6 py-4">
                          Fuzzy Score
                        </th>

                        <th className="px-6 py-4">
                          Size Ratio
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {result.detections.map((detection, index) => (

                        <tr
                          key={index}
                          className="border-t border-slate-800 transition hover:bg-slate-800/50"
                        >

                          <td className="px-6 py-4 text-slate-500">
                            {index + 1}
                          </td>


                          <td className="px-6 py-4">

                            <span className="font-semibold capitalize">
                              {detection.label}
                            </span>

                          </td>


                          <td className="px-6 py-4">

                            <span className="rounded-full bg-green-500/10 px-3 py-1 text-green-400">
                              {(detection.confidence * 100).toFixed(1)}%
                            </span>

                          </td>


                          <td className="px-6 py-4 text-green-400">
                            {detection.fuzzy_score.toFixed(2)}
                          </td>


                          <td className="px-6 py-4 text-slate-300">

                            {detection.size_ratio !== undefined
                              ? detection.size_ratio.toFixed(4)
                              : "—"}

                          </td>

                        </tr>

                      ))}

                    </tbody>

                  </table>

                </div>

              )}

            </div>


            {/* ==================================================
                PIPELINE
            ================================================== */}

            <div className="mt-8 rounded-3xl border border-slate-800 bg-slate-900 p-8">

              <h2 className="text-center text-3xl font-bold">
                🧠 Detection Pipeline
              </h2>

              <div className="mt-8 flex flex-wrap items-center justify-center gap-3">

                <Pipeline name="YOLOv8" />

                <Arrow />

                <Pipeline name="Fuzzy Logic" />

                <Arrow />

                <Pipeline name="Genetic Algorithm" />

                <Arrow />

                <Pipeline name="Hybrid Result" />

              </div>

            </div>

          </section>

        )}

      </div>


      {/* ==================================================
          FOOTER
      ================================================== */}

      <footer className="border-t border-slate-800 py-8 text-center text-sm text-slate-500">

        Hybrid Object Detection System
        <br />

        YOLO + Fuzzy Logic + Genetic Algorithm

      </footer>

    </main>
  );
}


/* ============================================================
   METRIC COMPONENT
============================================================ */

function Metric({
  title,
  value,
}: {
  title: string;
  value: number | string;
}) {

  return (

    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 text-center">

      <p className="text-sm text-slate-400">
        {title}
      </p>

      <p className="mt-3 text-4xl font-bold text-green-400">
        {value}
      </p>

    </div>

  );
}


/* ============================================================
   PIPELINE COMPONENT
============================================================ */

function Pipeline({
  name,
}: {
  name: string;
}) {

  return (

    <div className="rounded-xl border border-green-500/30 bg-green-500/10 px-5 py-3 font-semibold text-green-400">
      {name}
    </div>

  );
}


/* ============================================================
   ARROW COMPONENT
============================================================ */

function Arrow() {

  return (

    <span className="text-2xl text-slate-600">
      →
    </span>

  );

}