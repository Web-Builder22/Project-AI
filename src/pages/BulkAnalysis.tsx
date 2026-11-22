import { useCallback, useRef, useState } from 'react';
import {
  Upload,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Download,
  Play,
  FileText,
  BarChart2,
  RefreshCw,
} from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, LiveBadge, EmptyState } from '@/components/ui';
import { SentimentBadge } from '@/components/SentimentBadge';
import { analyzeBulkReviews, getBulkFileInfo } from '@/services/api';
import { SAMPLE_BULK_CSV_REVIEWS } from '@/data/realData';
import type { BulkAnalysisResult, BulkFileInfo } from '@/types';
import type { PageId } from '@/types/navigation';

interface BulkAnalysisProps {
  onNavigate: (page: PageId) => void;
}

function parseCsvPreview(
  file: File
): Promise<{ rowCount: number; hasReviewColumn: boolean; hasRatingColumn: boolean; sampleRows: string[] }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Could not read the file.'));
    reader.onload = () => {
      const content = String(reader.result ?? '');
      const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);
      const header = (lines[0] ?? '').toLowerCase();
      resolve({
        rowCount: Math.max(0, lines.length - 1),
        hasReviewColumn: /review|comment|text|feedback|summary/.test(header),
        hasRatingColumn: /rating|score|stars|rate/.test(header),
        sampleRows: lines.slice(1, 6),
      });
    };
    reader.readAsText(file.slice(0, 1024 * 1024));
  });
}

export default function BulkAnalysis({ onNavigate }: BulkAnalysisProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fileInfo, setFileInfo] = useState<BulkFileInfo | null>(null);
  const [activeFile, setActiveFile] = useState<File | null>(null);
  const [results, setResults] = useState<BulkAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const processFile = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeBulkReviews(file);
      setResults(res);
    } catch (err: any) {
      setError(err?.message || 'Bulk classification failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    setFileInfo(null);
    setResults(null);
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a .csv file containing customer reviews.');
      return;
    }
    setLoading(true);
    try {
      const { rowCount, hasReviewColumn, hasRatingColumn } = await parseCsvPreview(file);
      const info = getBulkFileInfo(file, rowCount, hasReviewColumn, hasRatingColumn);
      setFileInfo(info);
      setActiveFile(file);
      await processFile(file);
    } catch {
      setError('Could not read the file. Make sure it is a valid CSV.');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleLoadSampleBatch = async () => {
    setError(null);
    setLoading(true);
    setFileInfo({
      name: 'benchmark_sample_batch.csv',
      sizeBytes: 1850,
      rowCount: SAMPLE_BULK_CSV_REVIEWS.length,
      hasReviewColumn: true,
      hasRatingColumn: false,
    });
    try {
      const res = await analyzeBulkReviews(SAMPLE_BULK_CSV_REVIEWS);
      setResults(res);
    } catch (err: any) {
      setError(err?.message || 'Sample batch analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCsv = () => {
    if (!results || !results.downloadable_csv) return;
    const blob = new Blob([results.downloadable_csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'classified_reviews_output.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <AppShell page="bulk" onNavigate={onNavigate}>
      <PageHeader
        title="Bulk Review Analysis"
        subtitle="Batch classify thousands of customer reviews from CSV files with aggregate sentiment metrics and exportable results."
        icon={<Upload className="h-6 w-6" />}
        actions={<LiveBadge label="High-Throughput Inference" />}
      />

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Upload Review Dataset" subtitle="Supports CSV format with review text columns (up to 5,000 rows).">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const file = e.dataTransfer.files?.[0];
              if (file) void handleFile(file);
            }}
            onClick={() => inputRef.current?.click()}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
              dragOver
                ? 'border-brand-500 bg-brand-50'
                : 'border-slate-200 bg-slate-50 hover:border-brand-400 hover:bg-brand-50/40'
            }`}
          >
            {loading ? (
              <Loader2 className="mb-3 h-8 w-8 animate-spin text-brand-600" />
            ) : (
              <Upload className={`mb-3 h-8 w-8 ${dragOver ? 'text-brand-600' : 'text-slate-400'}`} />
            )}
            <p className="text-sm font-semibold text-slate-800">
              {loading ? 'Running ML batch classification…' : 'Drag & drop review CSV here'}
            </p>
            <p className="mt-1 text-xs text-slate-500">or click to browse local files</p>
            <p className="mt-3 text-[11px] text-slate-400">Accepted format: .csv</p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleFile(file);
              e.target.value = '';
            }}
          />

          <div className="mt-4 flex items-center justify-between border-t border-slate-100 pt-4">
            <span className="text-xs text-slate-500">Don't have a CSV handy?</span>
            <button
              onClick={handleLoadSampleBatch}
              disabled={loading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-semibold text-brand-700 hover:bg-brand-100 transition-colors"
            >
              <Play className="h-3.5 w-3.5" />
              Load Benchmark Test Batch (20 reviews)
            </button>
          </div>

          {error && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              {error}
            </div>
          )}
        </Card>

        <Card title="Dataset Ingestion Details" subtitle="Inspected schema and batch statistics." badge={<LiveBadge label="Automated Validation" />}>
          {!fileInfo ? (
            <EmptyState
              icon={<FileSpreadsheet className="h-6 w-6" />}
              title="No file processed yet"
              message="Upload a CSV or load the benchmark batch to view file telemetry and column detection."
            />
          ) : (
            <div className="animate-fade-in-up space-y-4">
              <div className="flex items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                <FileSpreadsheet className="h-8 w-8 shrink-0 text-emerald-600" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-bold text-slate-900">{fileInfo.name}</p>
                  <p className="text-xs text-slate-500">
                    {(fileInfo.sizeBytes / 1024).toFixed(1)} KB &bull; {fileInfo.rowCount.toLocaleString()} review rows
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="flex items-center gap-2 rounded-xl border border-slate-100 bg-slate-50 p-3">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                  <p className="text-xs font-medium text-slate-600">
                    Review column: {fileInfo.hasReviewColumn ? 'Detected' : 'Default text'}
                  </p>
                </div>
                <div className="flex items-center gap-2 rounded-xl border border-slate-100 bg-slate-50 p-3">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                  <p className="text-xs font-medium text-slate-600">
                    Leakage: Star rating decoupled
                  </p>
                </div>
              </div>

              {activeFile && (
                <button
                  onClick={() => processFile(activeFile)}
                  disabled={loading}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-brand-600 hover:text-brand-700"
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
                  Re-run classification
                </button>
              )}
            </div>
          )}
        </Card>
      </div>

      {results && (
        <Card
          className="mt-6 animate-fade-in-up"
          title="Batch Classification Results"
          subtitle={`Processed ${results.total_reviews} customer reviews with average confidence of ${(results.average_confidence * 100).toFixed(1)}%.`}
          badge={
            <button
              onClick={handleDownloadCsv}
              className="inline-flex items-center gap-2 rounded-xl bg-brand-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-brand-700 transition-colors"
            >
              <Download className="h-4 w-4" />
              Download Enriched CSV
            </button>
          }
        >
          {/* Summary Metric Cards */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 mb-6">
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Total Analyzed</p>
              <p className="mt-1 text-2xl font-bold text-slate-900">{results.total_reviews}</p>
            </div>
            <div className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-emerald-600">Positive</p>
              <p className="mt-1 text-2xl font-bold text-emerald-700">
                {results.counts.positive} <span className="text-sm font-normal">({results.percentages.positive}%)</span>
              </p>
            </div>
            <div className="rounded-xl border border-rose-100 bg-rose-50/50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-rose-600">Negative</p>
              <p className="mt-1 text-2xl font-bold text-rose-700">
                {results.counts.negative} <span className="text-sm font-normal">({results.percentages.negative}%)</span>
              </p>
            </div>
            <div className="rounded-xl border border-amber-100 bg-amber-50/50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-amber-600">Neutral</p>
              <p className="mt-1 text-2xl font-bold text-amber-700">
                {results.counts.neutral} <span className="text-sm font-normal">({results.percentages.neutral}%)</span>
              </p>
            </div>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-500 uppercase tracking-wide border-b border-slate-200">
                <tr>
                  <th className="py-3 px-4 w-12">#</th>
                  <th className="py-3 px-4">Review Text</th>
                  <th className="py-3 px-4 w-32">Predicted</th>
                  <th className="py-3 px-4 w-28 text-right">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {results.predictions.map((p) => (
                  <tr key={p.row_index} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-3 px-4 text-slate-400 font-mono">{p.row_index + 1}</td>
                    <td className="py-3 px-4 font-medium text-slate-800 max-w-md truncate">{p.review}</td>
                    <td className="py-3 px-4">
                      <SentimentBadge sentiment={p.predicted_sentiment} />
                    </td>
                    <td className="py-3 px-4 text-right font-bold text-slate-700">
                      {(p.confidence * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </AppShell>
  );
}
