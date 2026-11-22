import { useState } from 'react';
import { Search, Sparkles, Info, Loader2, Tag, AlertCircle, ShieldCheck } from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, LiveBadge } from '@/components/ui';
import { SentimentBadge } from '@/components/SentimentBadge';
import { analyzeSingleReview } from '@/services/api';
import type { AnalysisResult } from '@/types';
import type { PageId } from '@/types/navigation';

const EXAMPLES = [
  {
    label: 'High Praise',
    text: 'This headphones sound quality is fantastic and the battery lasts forever.',
    product: 'Noise Cancelling Wireless Headphones (Black)',
  },
  {
    label: 'Severe Defect',
    text: 'Arrived two weeks late and the box was crushed. Defective unit stopped working on day 2. Very disappointed.',
    product: 'Smart Watch X2',
  },
  {
    label: 'Minor Praise / Terse',
    text: 'Good product for this price tag, neat build and fast shipping.',
    product: 'Nova Dry Iron (Grey)',
  },
  {
    label: 'Subtle Neutral',
    text: 'The monitor is okay for basic spreadsheets and emails. Nothing special to write home about.',
    product: 'Ultra HD 4K Monitor',
  },
  {
    label: 'Negation Case',
    text: 'Not good at all, volume is too low and bass is virtually non existent.',
    product: 'Bluetooth Speaker Mini',
  },
];

interface SingleAnalysisProps {
  onNavigate: (page: PageId) => void;
}

export default function SingleAnalysis({ onNavigate }: SingleAnalysisProps) {
  const [text, setText] = useState('');
  const [productName, setProductName] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!text.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await analyzeSingleReview(text, productName);
      setResult(res);
    } catch (err: any) {
      console.error(err);
      setError(
        err?.message ||
          'Live prediction requires the ML microservice running on port 8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  const confidenceColor = (c: number) =>
    c >= 80 ? 'bg-emerald-500' : c >= 50 ? 'bg-amber-500' : 'bg-slate-400';

  return (
    <AppShell page="single" onNavigate={onNavigate}>
      <PageHeader
        title="Single Review Analysis"
        subtitle="Real-time sentiment classification using the deployed TF-IDF scikit-learn best model pipeline."
        icon={<Search className="h-6 w-6" />}
        actions={<LiveBadge label="Production Inference Active" />}
      />

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Customer Review Input" subtitle="Paste or type customer feedback to inspect sentiment, confidence, and keyword attribution.">
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500 mb-1.5">
                Product Name (Optional)
              </label>
              <input
                type="text"
                value={productName}
                onChange={(e) => setProductName(e.target.value)}
                placeholder="e.g. Noise Cancelling Wireless Headphones (Black)"
                className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-brand-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-brand-100"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500 mb-1.5">
                Review Text
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="e.g. Sound quality exceeded my expectations, seamless bluetooth connection and fast delivery."
                rows={6}
                className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm leading-relaxed text-slate-800 placeholder:text-slate-400 focus:border-brand-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-brand-100"
              />
            </div>
          </div>

          <div className="mt-4">
            <span className="text-xs font-medium text-slate-400">Quick benchmark examples:</span>
            <div className="mt-2 flex flex-wrap gap-2">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setText(ex.text);
                    setProductName(ex.product);
                  }}
                  className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600 transition-colors hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700"
                >
                  {ex.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-6 flex items-center justify-between gap-4 border-t border-slate-100 pt-4">
            <p className="text-xs text-slate-400">{text.trim().length.toLocaleString()} characters</p>
            <button
              onClick={handleAnalyze}
              disabled={!text.trim() || loading}
              className="inline-flex items-center gap-2 rounded-xl bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              {loading ? 'Classifying…' : 'Classify Sentiment'}
            </button>
          </div>
        </Card>

        <Card
          title="Classification Telemetry"
          subtitle="Model predictions, 3-class probability distribution, and lexical attribution."
          badge={<LiveBadge label="Trained Pipeline" />}
        >
          {loading && (
            <div className="flex animate-pulse flex-col gap-4 py-8">
              <div className="h-10 w-36 rounded-lg bg-slate-100" />
              <div className="h-4 w-full rounded-full bg-slate-100" />
              <div className="h-24 w-full rounded-xl bg-slate-100" />
            </div>
          )}

          {!loading && error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 p-5 text-xs text-rose-800 space-y-3">
              <div className="flex items-center gap-2 font-bold text-rose-900">
                <AlertCircle className="h-4 w-4 text-rose-600 shrink-0" />
                <span>Live ML Microservice Unreachable</span>
              </div>
              <p className="leading-relaxed">{error}</p>
              <div className="rounded-lg bg-white/80 p-3 font-mono text-[11px] text-slate-700 border border-rose-100">
                python -m uvicorn main:app --host 0.0.0.0 --port 8000
              </div>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 px-6 py-14 text-center">
              <Sparkles className="mb-3 h-8 w-8 text-slate-300" />
              <p className="text-sm font-semibold text-slate-700">Awaiting input review</p>
              <p className="mt-1 max-w-xs text-xs text-slate-500">
                Choose an example or enter a customer review on the left, then click Classify Sentiment.
              </p>
            </div>
          )}

          {!loading && result && (
            <div className="animate-fade-in-up space-y-5">
              <div className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 p-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Classified Sentiment</p>
                  <div className="mt-2">
                    <SentimentBadge sentiment={result.sentiment} />
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Model Confidence</p>
                  <p className="mt-1 text-3xl font-bold text-slate-900">
                    {result.confidence.toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* 3-Class Probability Bars */}
              {result.probabilities && (
                <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 space-y-2.5">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Posterior Probability Distribution
                  </p>
                  <div className="space-y-2">
                    {[
                      {
                        label: 'Positive',
                        prob: (result.probabilities.positive || 0) * 100,
                        barColor: 'bg-emerald-500',
                        textColor: 'text-emerald-700',
                      },
                      {
                        label: 'Neutral',
                        prob: (result.probabilities.neutral || 0) * 100,
                        barColor: 'bg-amber-500',
                        textColor: 'text-amber-700',
                      },
                      {
                        label: 'Negative',
                        prob: (result.probabilities.negative || 0) * 100,
                        barColor: 'bg-rose-500',
                        textColor: 'text-rose-700',
                      },
                    ].map((item) => (
                      <div key={item.label} className="text-xs">
                        <div className="flex justify-between font-medium text-slate-600 mb-1">
                          <span>{item.label}</span>
                          <span className={`font-bold ${item.textColor}`}>{item.prob.toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
                          <div
                            className={`h-full ${item.barColor} transition-all duration-500`}
                            style={{ width: `${item.prob}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Contributing Words / Attribution */}
              {result.top_contributing_words && result.top_contributing_words.length > 0 && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-2">
                    Feature Attribution (Top Learned Tokens)
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {result.top_contributing_words.map((w, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-brand-200 bg-brand-50 px-2.5 py-1 text-xs font-semibold text-brand-800"
                      >
                        <Tag className="h-3 w-3 text-brand-500" />
                        {w.word}
                        <span className="text-[10px] text-brand-600">({w.weight > 0 ? `+${w.weight.toFixed(2)}` : w.weight.toFixed(2)})</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {result.explanation && (
                <div className="flex gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <Info className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" />
                  <p className="text-xs leading-relaxed text-slate-700">{result.explanation}</p>
                </div>
              )}

              <div className="flex items-center gap-2 text-xs text-slate-400">
                <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
                Pipeline: TF-IDF (10,000 unigram/bigram vocabulary) &rarr; Calibrated Linear Model
              </div>
            </div>
          )}
        </Card>
      </div>
    </AppShell>
  );
}
