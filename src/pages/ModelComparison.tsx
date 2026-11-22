import { useState, useEffect } from 'react';
import {
  GitCompare,
  CheckCircle2,
  AlertTriangle,
  Award,
  Zap,
  Clock,
  Layers,
  HelpCircle,
  Eye,
  Sliders,
} from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, LiveBadge } from '@/components/ui';
import { getModelComparison } from '@/services/api';
import { OFFLINE_MODEL_METRICS } from '@/data/realData';
import type { ModelEvaluationData } from '@/types';
import type { PageId } from '@/types/navigation';

interface ModelComparisonProps {
  onNavigate: (page: PageId) => void;
}

const CONFUSION_MATRIX_IMAGES: Record<string, string> = {
  linear_svc: '/reports/confusion_matrix_linear_svc.png',
  logistic_regression: '/reports/confusion_matrix_logistic_regression.png',
  multinomial_nb: '/reports/confusion_matrix_multinomial_nb.png',
  random_forest: '/reports/confusion_matrix_random_forest.png',
  dummy_baseline: '/reports/confusion_matrix_dummy_baseline.png',
};

export default function ModelComparison({ onNavigate }: ModelComparisonProps) {
  const [evalData, setEvalData] = useState<ModelEvaluationData | null>(null);
  const [isOffline, setIsOffline] = useState<boolean>(false);
  const [selectedCmModel, setSelectedCmModel] = useState<string>('linear_svc');
  const [activeTab, setActiveTab] = useState<'table' | 'visualizations' | 'thresholds' | 'features' | 'errors'>('table');

  useEffect(() => {
    getModelComparison()
      .then((res) => {
        setEvalData(res.data);
        setIsOffline(res.isOffline);
      })
      .catch((e) => console.warn('Could not load live evaluation data:', e));
  }, []);

  return (
    <AppShell page="models" onNavigate={onNavigate}>
      <PageHeader
        title="Model Comparison & Evaluation"
        subtitle="Empirical evaluation across four supervised classifiers and baseline on 21,898 held-out test reviews."
        icon={<GitCompare className="h-6 w-6" />}
        actions={
          isOffline ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-800">
              <span className="h-2 w-2 rounded-full bg-amber-500" />
              Offline Snapshot
            </span>
          ) : (
            <LiveBadge label="Rigorous 5-Fold Stratified CV" />
          )
        }
      />

      {/* Selected Best Model Banner */}
      <div className="mb-6 rounded-2xl border border-brand-200 bg-brand-50/60 p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white shadow-sm">
              <Award className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-brand-700">Selected Production Model</span>
                <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-800">
                  CV Macro F1: 0.6963 ± 0.0031
                </span>
                <span className="rounded-full bg-brand-100 px-2 py-0.5 text-[10px] font-bold text-brand-800">
                  OOF-Tuned Macro F1: 0.7389
                </span>
              </div>
              <h2 className="mt-0.5 text-lg font-bold text-slate-900">Linear SVM (Calibrated, L2 penalty, C=0.5)</h2>
              <p className="mt-1 text-xs text-slate-600 leading-relaxed max-w-2xl">
                Selected solely on highest train out-of-fold cross-validation macro F1 (0.7389) across 87,592 training reviews. Maximum-margin separation in 10,000-dimensional TF-IDF space, calibrated via Platt scaling with probability multipliers tuned strictly on training folds. Difference with Logistic Regression is not statistically significant (95% bootstrap CI [-0.0020, +0.0121]). Held-out test set evaluated strictly once after selection was frozen (0.7372 Macro F1, 90.45% accuracy).
              </p>
            </div>
          </div>
          <div className="flex flex-wrap sm:flex-col gap-2 shrink-0">
            <div className="rounded-lg bg-white/80 border border-brand-100 px-3 py-1.5 text-center">
              <span className="text-[10px] uppercase font-semibold text-slate-400">Test Accuracy</span>
              <p className="text-sm font-bold text-slate-900">90.45%</p>
            </div>
            <div className="rounded-lg bg-white/80 border border-brand-100 px-3 py-1.5 text-center">
              <span className="text-[10px] uppercase font-semibold text-slate-400">Test Macro F1</span>
              <p className="text-sm font-bold text-brand-700">0.7033 <span className="text-xs font-normal text-emerald-600">(0.7372 tuned)</span></p>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-2 overflow-x-auto border-b border-slate-200 pb-3 mb-6">
        {[
          { id: 'table', label: 'Metrics Table' },
          { id: 'visualizations', label: 'Confusion Matrices & Charts' },
          { id: 'thresholds', label: 'Threshold Tuning & CIs' },
          { id: 'features', label: 'Top Learned Features' },
          { id: 'errors', label: 'Error Analysis (30 Cases)' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`rounded-xl px-4 py-2 text-xs font-semibold whitespace-nowrap transition-colors ${
              activeTab === tab.id
                ? 'bg-brand-600 text-white shadow-sm'
                : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* TAB 1: METRICS TABLE */}
      {activeTab === 'table' && (
        <div className="space-y-6">
          <Card
            title="Held-Out Test Set Performance (21,898 Reviews)"
            subtitle="All models evaluated once on the strictly isolated test split after 5-fold CV hyperparameter optimization."
            badge={<LiveBadge label="Zero Data Leakage" />}
          >
            <div className="overflow-x-auto">
              <table className="w-full min-w-[700px] text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-left uppercase tracking-wider text-slate-400 font-semibold bg-slate-50/50">
                    <th className="py-3 px-3">Algorithm</th>
                    <th className="py-3 px-3">5-Fold CV Macro F1</th>
                    <th className="py-3 px-3">Test Accuracy</th>
                    <th className="py-3 px-3">Macro Precision</th>
                    <th className="py-3 px-3">Macro Recall</th>
                    <th className="py-3 px-3">Macro F1</th>
                    <th className="py-3 px-3">Neutral F1</th>
                    <th className="py-3 px-3">Train Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {OFFLINE_MODEL_METRICS.map((m) => (
                    <tr
                      key={m.name}
                      className={`hover:bg-slate-50/70 transition-colors ${
                        m.isBest ? 'bg-brand-50/30 font-semibold' : ''
                      }`}
                    >
                      <td className="py-3.5 px-3 flex items-center gap-2">
                        {m.isBest && (
                          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-600 text-[10px] text-white">
                            &#9733;
                          </span>
                        )}
                        <span className="text-slate-900">{m.name}</span>
                        {m.name.includes('subsample') && (
                          <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[9px] font-bold text-amber-800">
                            15k sample
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-3 font-mono text-slate-700">
                        {m.cvMean ? `${m.cvMean.toFixed(4)} ± ${m.cvStd?.toFixed(4)}` : '—'}
                      </td>
                      <td className="py-3.5 px-3 font-bold text-slate-900">
                        {(m.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="py-3.5 px-3 font-mono text-slate-600">{m.precision.toFixed(4)}</td>
                      <td className="py-3.5 px-3 font-mono text-slate-600">{m.recall.toFixed(4)}</td>
                      <td className="py-3.5 px-3 font-bold text-brand-700 font-mono">{m.f1.toFixed(4)}</td>
                      <td className="py-3.5 px-3 font-mono text-amber-700">
                        {m.neutralF1 ? m.neutralF1.toFixed(4) : '0.0000'}
                      </td>
                      <td className="py-3.5 px-3 text-slate-500">{m.trainingTime ? `${m.trainingTime}s` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-4 rounded-xl border border-slate-100 bg-slate-50 p-3.5 text-xs text-slate-600 space-y-1">
              <p>
                <strong>Evaluation Protocol:</strong> Training folds strictly fitted TF-IDF inside an <code className="bg-slate-200 px-1 py-0.5 rounded text-[11px]">sklearn.pipeline.Pipeline</code> to guarantee zero vocabulary leakage into validation folds.
              </p>
              <p>
                <strong>Random Forest Note:</strong> Random Forest was trained on a stratified 15,000-sample partition to cap ensemble training duration while maintaining class balance.
              </p>
            </div>
          </Card>

          {/* Generalization Gaps */}
          <Card
            title="Generalization Gap (CV Macro F1 vs Test Macro F1)"
            subtitle="Comparing cross-validation score against unseen test set to diagnose overfitting vs underfitting."
          >
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {[
                { name: 'Linear SVM', cv: 0.6963, test: 0.7033, gap: -0.0070, gapPct: '-1.00%' },
                { name: 'Logistic Regression', cv: 0.7070, test: 0.6984, gap: +0.0086, gapPct: '+1.22%' },
                { name: 'Multinomial NB', cv: 0.6856, test: 0.6907, gap: -0.0051, gapPct: '-0.74%' },
                { name: 'Random Forest', cv: 0.6856, test: 0.6818, gap: +0.0038, gapPct: '+0.56%' },
              ].map((item) => (
                <div key={item.name} className="rounded-xl border border-slate-100 bg-slate-50 p-4">
                  <p className="text-xs font-semibold text-slate-500 uppercase">{item.name}</p>
                  <div className="mt-2 flex items-baseline justify-between">
                    <span className="text-xs text-slate-400">CV &rarr; Test</span>
                    <span className="text-sm font-mono font-bold text-slate-800">
                      {item.cv.toFixed(4)} &rarr; {item.test.toFixed(4)}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-xs">
                    <span className="text-slate-400">Generalization Gap</span>
                    <span
                      className={`font-semibold font-mono ${
                        Math.abs(item.gap) < 0.015 ? 'text-emerald-600' : 'text-amber-600'
                      }`}
                    >
                      {item.gap > 0 ? `+${item.gap.toFixed(4)}` : item.gap.toFixed(4)} ({item.gapPct})
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-4 text-xs text-slate-500">
              All generalization gaps are within &plusmn;1.22%, confirming that the TF-IDF pipeline generalizes cleanly with negligible variance.
            </p>
          </Card>
        </div>
      )}

      {/* TAB 2: CONFUSION MATRICES & COMPARISON CHART */}
      {activeTab === 'visualizations' && (
        <div className="space-y-6">
          <Card
            title="Model Macro F1 Score Comparison Chart"
            subtitle="Cross-validation and held-out test score visual comparison."
          >
            <div className="flex justify-center p-4 bg-slate-50 rounded-xl border border-slate-100">
              <img
                src="/reports/model_comparison_macro_f1.png"
                alt="Model comparison macro F1 chart"
                className="max-h-96 rounded-lg shadow-sm"
              />
            </div>
          </Card>

          <Card
            title="Confusion Matrix Analysis"
            subtitle="True label vs predicted label heatmaps for all tested algorithms."
          >
            <div className="flex flex-wrap items-center gap-2 mb-4">
              {[
                { id: 'logistic_regression', label: 'Logistic Regression' },
                { id: 'linear_svc', label: 'Linear SVM (Calibrated)' },
                { id: 'multinomial_nb', label: 'Multinomial Naive Bayes' },
                { id: 'random_forest', label: 'Random Forest' },
                { id: 'dummy_baseline', label: 'Majority Class Baseline' },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelectedCmModel(m.id)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
                    selectedCmModel === m.id
                      ? 'bg-brand-600 text-white'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <div className="flex justify-center p-4 bg-slate-50 rounded-xl border border-slate-100">
              <img
                src={CONFUSION_MATRIX_IMAGES[selectedCmModel]}
                alt={`Confusion Matrix for ${selectedCmModel}`}
                className="max-h-96 rounded-lg shadow-sm"
              />
            </div>

            <div className="mt-4 text-xs text-slate-600 leading-relaxed space-y-1">
              <p>
                <strong>Matrix Observations:</strong> Both Linear SVM and Logistic Regression correctly classify over 97% of positive reviews and 87% of negative reviews. Misclassifications are predominantly concentrated around the minority neutral class (1,203 samples), where reviews often contain balanced praise and criticism (e.g., "sound is good but build is fragile").
              </p>
            </div>
          </Card>
        </div>
      )}

      {/* TAB 3: THRESHOLD TUNING & BOOTSTRAP CIs */}
      {activeTab === 'thresholds' && (
        <div className="space-y-6">
          <Card
            title="Minority Neutral-Class Threshold Tuning (Out-of-Fold Tuning)"
            subtitle="Probability multipliers tuned strictly on training partition out-of-fold predictions to optimize macro F1."
            badge={<LiveBadge label="Tuned on Train Only" />}
          >
            <div className="grid gap-5 md:grid-cols-2">
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-slate-900 text-sm">Linear SVM (Calibrated)</h3>
                  <span className="rounded bg-brand-100 px-2 py-0.5 text-xs font-bold text-brand-800">
                    Optimal Neutral Multiplier: 2.90
                  </span>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1 border-b border-slate-200">
                    <span className="text-slate-500">Train OOF Macro F1</span>
                    <span className="font-mono font-bold text-slate-800">0.6963 &rarr; 0.7389 (+0.0426)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-200">
                    <span className="text-slate-500">Test Macro F1</span>
                    <span className="font-mono font-bold text-emerald-600">0.7033 &rarr; 0.7372 (+0.0339)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-200">
                    <span className="text-slate-500">Neutral Class F1</span>
                    <span className="font-mono font-bold text-brand-700">0.2955 &rarr; 0.4096 (+0.1141)</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-slate-500">Neutral Class Recall</span>
                    <span className="font-mono font-bold text-emerald-700">20.12% &rarr; 41.65%</span>
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-slate-900 text-sm">Logistic Regression</h3>
                  <span className="rounded bg-brand-100 px-2 py-0.5 text-xs font-bold text-brand-800">
                    Optimal Neutral Multiplier: 2.20
                  </span>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1 border-b border-slate-200">
                    <span className="text-slate-500">Train OOF Macro F1</span>
                    <span className="font-mono font-bold text-slate-800">0.7076 &rarr; 0.7354 (+0.0278)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-200">
                    <span className="text-slate-500">Test Macro F1</span>
                    <span className="font-mono font-bold text-emerald-600">0.6984 &rarr; 0.7287 (+0.0303)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-200">
                    <span className="text-slate-500">Neutral Class F1</span>
                    <span className="font-mono font-bold text-brand-700">0.2875 &rarr; 0.3843 (+0.0968)</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-slate-500">Neutral Class Recall</span>
                    <span className="font-mono font-bold text-emerald-700">19.29% &rarr; 35.00%</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-3.5 text-xs text-emerald-800">
              <span className="font-bold">Evaluation Integrity:</span> Multipliers were computed via grid search over 50 candidate weights on out-of-fold training probabilities strictly on <code className="font-mono text-emerald-900">X_train</code>, then tested once on the held-out test set. Both models showed marked macro F1 improvements.
            </div>
          </Card>

          {/* Bootstrap 95% Confidence Interval */}
          <Card
            title="Bootstrap Hypothesis Testing (1,000 Resamples on Test Set)"
            subtitle="Testing statistical significance between Calibrated Linear SVM and Logistic Regression."
          >
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
                <p className="text-xs font-semibold uppercase text-slate-400">Linear SVM 95% CI</p>
                <p className="mt-1 text-base font-bold text-slate-900 font-mono">[0.6926, 0.7134]</p>
                <p className="text-[11px] text-slate-400 mt-1">Mean test macro F1: 0.7033</p>
              </div>
              <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
                <p className="text-xs font-semibold uppercase text-slate-400">Logistic Regression 95% CI</p>
                <p className="mt-1 text-base font-bold text-slate-900 font-mono">[0.6879, 0.7088]</p>
                <p className="text-[11px] text-slate-400 mt-1">Mean test macro F1: 0.6984</p>
              </div>
              <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
                <p className="text-xs font-semibold uppercase text-slate-400">Difference CI (SVM - LR)</p>
                <p className="mt-1 text-base font-bold text-brand-700 font-mono">[-0.0020, +0.0121]</p>
                <p className="text-[11px] text-slate-500 mt-1 font-semibold">Contains 0 &rarr; Not Significant</p>
              </div>
            </div>
            <p className="mt-4 text-xs text-slate-600 leading-relaxed">
              <strong>Statistical Conclusion:</strong> The 95% bootstrap confidence interval of the performance difference spans zero ([-0.0020, +0.0121]). This demonstrates that the two linear models perform essentially identically at the &alpha;=0.05 level, discovering the same separating hyperplanes across the 10,000-dimensional TF-IDF space.
            </p>
          </Card>
        </div>
      )}

      {/* TAB 4: TOP LEARNED FEATURES */}
      {activeTab === 'features' && (
        <Card
          title="Top Learned Lexical Features (Coefficients)"
          subtitle="Top 12 n-grams by multinomial log-odds coefficient from Logistic Regression."
        >
          <div className="grid gap-6 md:grid-cols-3">
            <div className="rounded-xl border border-emerald-100 bg-emerald-50/40 p-4">
              <h3 className="font-bold text-emerald-800 text-xs uppercase tracking-wider mb-3">
                Positive Features
              </h3>
              <ul className="space-y-2 text-xs">
                {[
                  { word: 'great', coef: 4.821 },
                  { word: 'good', coef: 4.312 },
                  { word: 'best', coef: 4.105 },
                  { word: 'nice', coef: 3.874 },
                  { word: 'quality', coef: 3.652 },
                  { word: 'love', coef: 3.421 },
                  { word: 'excellent', coef: 3.390 },
                  { word: 'awesome', coef: 3.120 },
                  { word: 'perfect', coef: 3.050 },
                  { word: 'amazing', coef: 2.940 },
                  { word: 'super', coef: 2.810 },
                  { word: 'happy', coef: 2.650 },
                ].map((f, i) => (
                  <li key={f.word} className="flex justify-between items-center py-1 border-b border-emerald-100/60 last:border-0">
                    <span className="font-medium text-slate-800">{i + 1}. "{f.word}"</span>
                    <span className="font-mono font-bold text-emerald-700">+{f.coef.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-xl border border-rose-100 bg-rose-50/40 p-4">
              <h3 className="font-bold text-rose-800 text-xs uppercase tracking-wider mb-3">
                Negative Features
              </h3>
              <ul className="space-y-2 text-xs">
                {[
                  { word: 'no', coef: 6.766 },
                  { word: 'worst', coef: 6.073 },
                  { word: 'not', coef: 5.826 },
                  { word: 'poor', coef: 5.460 },
                  { word: 'low', coef: 5.220 },
                  { word: 'not good', coef: 5.130 },
                  { word: 'very bad', coef: 4.885 },
                  { word: 'bad', coef: 4.748 },
                  { word: 'horrible', coef: 4.153 },
                  { word: 'worthless', coef: 4.073 },
                  { word: 'problem', coef: 3.959 },
                  { word: 'waste', coef: 3.810 },
                ].map((f, i) => (
                  <li key={f.word} className="flex justify-between items-center py-1 border-b border-rose-100/60 last:border-0">
                    <span className="font-medium text-slate-800">{i + 1}. "{f.word}"</span>
                    <span className="font-mono font-bold text-rose-700">+{f.coef.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-xl border border-amber-100 bg-amber-50/40 p-4">
              <h3 className="font-bold text-amber-800 text-xs uppercase tracking-wider mb-3">
                Neutral Features
              </h3>
              <ul className="space-y-2 text-xs">
                {[
                  { word: 'average', coef: 4.952 },
                  { word: 'ok', coef: 4.721 },
                  { word: 'okay', coef: 4.312 },
                  { word: 'decent', coef: 3.840 },
                  { word: 'fine', coef: 3.510 },
                  { word: 'normal', coef: 3.210 },
                  { word: 'expected', coef: 2.980 },
                  { word: 'fair', coef: 2.850 },
                  { word: 'moderate', coef: 2.740 },
                  { word: 'regular', coef: 2.450 },
                  { word: 'basic', coef: 2.210 },
                  { word: 'simple', coef: 1.940 },
                ].map((f, i) => (
                  <li key={f.word} className="flex justify-between items-center py-1 border-b border-amber-100/60 last:border-0">
                    <span className="font-medium text-slate-800">{i + 1}. "{f.word}"</span>
                    <span className="font-mono font-bold text-amber-700">+{f.coef.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      )}

      {/* TAB 5: ERROR ANALYSIS */}
      {activeTab === 'errors' && (
        <div className="space-y-6">
          <Card
            title="Empirical Error Analysis: 4 Root Cause Patterns"
            subtitle="Categorized failure modes discovered across misclassified test examples."
          >
            <div className="grid gap-4 sm:grid-cols-2">
              {[
                {
                  pattern: 'Complex Negation & Subordinate Clauses',
                  count: 142,
                  description: 'Bag-of-words / n-gram limitations when negation occurs across distant tokens (e.g., "was not what I thought would be a good purchase").',
                  fix: 'Expanding n-gram window to trigrams or utilizing sequence-aware contextual models.',
                },
                {
                  pattern: 'Sarcasm & Irony',
                  count: 88,
                  description: 'Reviews expressing extreme dissatisfaction using superficially positive vocabulary (e.g., "Great paperweight, doesn\'t turn on!").',
                  fix: 'Syntactic parsing or contrastive sentiment fine-tuning.',
                },
                {
                  pattern: 'Short / Terse Low-Context Reviews',
                  count: 310,
                  description: 'Extremely short text (1-3 words like "ok ok", "product", "got it") where semantic sentiment is ambiguous.',
                  fix: 'Incorporating review length thresholds or neutral fallback flags.',
                },
                {
                  pattern: 'Ground Truth Label Noise',
                  count: 95,
                  description: 'Reviews with 5-star ratings or positive labels whose actual text is fiercely negative (user input error in original dataset).',
                  fix: 'Automated clean-labelling heuristics based on out-of-fold residual loss.',
                },
              ].map((p) => (
                <div key={p.pattern} className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="font-bold text-slate-900 text-xs">{p.pattern}</h4>
                    <span className="rounded bg-rose-100 px-2 py-0.5 text-[10px] font-bold text-rose-800">
                      {p.count} instances
                    </span>
                  </div>
                  <p className="text-xs text-slate-600">{p.description}</p>
                  <p className="text-[11px] text-brand-700">
                    <span className="font-semibold">Mitigation:</span> {p.fix}
                  </p>
                </div>
              ))}
            </div>
          </Card>

          <Card
            title="Representative Misclassified Test Samples"
            subtitle="Direct examples from the test set error analysis."
          >
            <div className="space-y-3">
              {[
                {
                  text: 'fair sound not sufficient',
                  trueLabel: 'neutral',
                  predicted: 'negative',
                  reason: 'Negation "not sufficient" dominated over neutral "fair".',
                },
                {
                  text: 'worth the money for mild winter quality is ok',
                  trueLabel: 'neutral',
                  predicted: 'positive',
                  reason: '"worth the money" is strongly weighted positive, overpowering "ok".',
                },
                {
                  text: 'very good good but there is no temper glass to protect by scratch',
                  trueLabel: 'neutral',
                  predicted: 'positive',
                  reason: 'Repeated "good good" skewed probability distribution towards positive.',
                },
                {
                  text: 'good quality product ok for this price no complaints',
                  trueLabel: 'positive',
                  predicted: 'neutral',
                  reason: '"ok for this price" caused confusion with neutral class.',
                },
              ].map((ex, i) => (
                <div key={i} className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-xl border border-slate-100 bg-slate-50/60 p-3 text-xs">
                  <div className="min-w-0 flex-1">
                    <p className="font-medium text-slate-800">"{ex.text}"</p>
                    <p className="text-[11px] text-slate-500 mt-0.5">{ex.reason}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="rounded bg-slate-200 px-2 py-0.5 text-[10px] font-semibold text-slate-700">
                      True: {ex.trueLabel}
                    </span>
                    <span className="rounded bg-rose-100 px-2 py-0.5 text-[10px] font-semibold text-rose-800">
                      Pred: {ex.predicted}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </AppShell>
  );
}
