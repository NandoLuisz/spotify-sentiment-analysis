import { useState } from "react";

interface PredictionResponse {
  review: string;
  sentiment: string;
  probabilities: Record<string, number>;
}

export function App() {
  const [review, setReview] = useState<string>("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review }),
      });

      if (!response.ok) {
        throw new Error("Failed to classify review.");
      }

      const data: PredictionResponse = await response.json();

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An unexpected error occurred.",
      );
    } finally {
      setLoading(false);
    }
  }

  const sentimentColors: Record<string, string> = {
    Positivo: "#1DB954",
    Negativo: "#F87171",
    Neutro: "#9CA3AF",
  };

  return (
    <div className="min-h-screen bg-[#121212] text-white flex items-center justify-center px-6 py-12">
      <main className="w-full max-w-2xl">
        <header className="mb-10 text-center">
          <h1 className="text-4xl font-bold tracking-tight">
            Spotify <span className="text-[#1DB954]">Sentiment</span>
          </h1>

          <p className="mt-3 text-gray-400">
            Discover the sentiment behind your Spotify reviews.
          </p>
        </header>

        <div className="rounded-2xl bg-[#181818] p-8 shadow-2xl border border-[#282828]">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label
                htmlFor="review"
                className="mb-3 block text-sm font-medium text-gray-200"
              >
                Your review
              </label>

              <textarea
                id="review"
                value={review}
                // onChange={(event) => {
                //   const text = event.target.value.replace(/[^a-zA-Z\s]/g, "");

                //   setReview(text);
                //   setResult(null);
                //   setError("");
                // }}
                onChange={(event) => {
                  setReview(event.target.value);
                  setResult(null);
                  setError("");
                }}
                placeholder="Tell us what you think about Spotify..."
                className="w-full min-h-40 resize-y rounded-xl border border-[#333] bg-[#242424] p-4 text-white placeholder-gray-500 outline-none transition focus:border-[#1DB954] focus:ring-2 focus:ring-[#1DB954]/20"
                required
              />
            </div>

            <button
              type="submit"
              disabled={!review.trim() || loading}
              className="w-full rounded-full bg-[#1DB954] px-6 py-3 font-bold text-black transition hover:bg-[#1ed760] hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100"
            >
              {loading ? "Analyzing..." : "Analyze Sentiment"}
            </button>
          </form>

          {error && (
            <div className="mt-6 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-400">
              {error}
            </div>
          )}

          <section className="mt-8 rounded-xl border border-[#282828] bg-[#222222] p-5">
            <h2 className="mb-4 text-lg font-semibold">Analysis result</h2>

            {!result && !loading && !error && (
              <p className="text-gray-400">
                Your sentiment result will appear here.
              </p>
            )}

            {loading && (
              <p className="animate-pulse text-gray-400">
                Analyzing your review...
              </p>
            )}

            {result && (
              <div className="space-y-6">
                <div>
                  <p className="mb-2 text-sm text-gray-400">
                    Predicted sentiment
                  </p>

                  <span
                    className="text-2xl font-bold"
                    style={{
                      color: sentimentColors[result.sentiment] ?? "#FFFFFF",
                    }}
                  >
                    {result.sentiment}
                  </span>
                </div>

                <div>
                  <h3 className="mb-4 text-sm font-medium text-gray-300">
                    Class probabilities
                  </h3>

                  <div className="space-y-4">
                    {Object.entries(result.probabilities).map(
                      ([sentiment, probability]) => (
                        <div key={sentiment}>
                          <div className="mb-2 flex justify-between text-sm">
                            <span className="text-gray-300">{sentiment}</span>

                            <span className="font-medium text-white">
                              {probability.toFixed(2)}%
                            </span>
                          </div>

                          <div className="h-2 overflow-hidden rounded-full bg-[#333333]">
                            <div
                              className="h-full rounded-full transition-all duration-500"
                              style={{
                                width: `${probability}%`,
                                backgroundColor:
                                  sentimentColors[sentiment] ?? "#FFFFFF",
                              }}
                            />
                          </div>
                        </div>
                      ),
                    )}
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>

        <footer className="mt-6 text-center text-sm text-gray-500">
          Powered by Machine Learning & Multinomial Naive Bayes
        </footer>
      </main>
    </div>
  );
}
