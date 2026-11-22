import { useState, useEffect } from 'react';
import {
  Lightbulb,
  ThumbsUp,
  ThumbsDown,
  TrendingUp,
  Package,
  AlertTriangle,
  FileText,
  Star,
  CheckCircle2,
} from 'lucide-react';
import AppShell from '@/components/AppShell';
import PageHeader from '@/components/PageHeader';
import { Card, LiveBadge } from '@/components/ui';
import { POSITIVE_TOPICS, COMMON_COMPLAINTS, FEEDBACK_TRENDS, REAL_TOP_PRODUCTS, REAL_KEY_FINDINGS } from '@/data/realData';
import { getEDASummary } from '@/services/api';
import type { EDASummaryData } from '@/types';
import type { PageId } from '@/types/navigation';

interface CustomerInsightsProps {
  onNavigate: (page: PageId) => void;
}

function TopicList({
  items,
  color,
  icon,
}: {
  items: { topic: string; count: number; share: number }[];
  color: string;
  icon: 'up' | 'down';
}) {
  const Icon = icon === 'up' ? ThumbsUp : ThumbsDown;
  const max = items[0]?.count || 1;
  return (
    <ul className="space-y-4">
      {items.map((item) => (
        <li key={item.topic}>
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <Icon className={`h-4 w-4 shrink-0 ${icon === 'up' ? 'text-emerald-500' : 'text-rose-500'}`} />
              <p className="text-xs font-semibold text-slate-800">{item.topic}</p>
            </div>
            <p className="shrink-0 text-xs font-semibold text-slate-500">
              {item.count.toLocaleString()} ({Math.round(item.share * 100)}%)
            </p>
          </div>
          <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-slate-100">
            <div
              className={`h-full rounded-full ${color} transition-all duration-500`}
              style={{ width: `${(item.count / max) * 100}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function CustomerInsights({ onNavigate }: CustomerInsightsProps) {
  const [edaData, setEdaData] = useState<EDASummaryData | null>(null);
  const [isOffline, setIsOffline] = useState<boolean>(false);

  useEffect(() => {
    getEDASummary()
      .then((res) => {
        setEdaData(res.data);
        setIsOffline(res.isOffline);
      })
      .catch((e) => {
        console.warn('Could not load live EDA:', e);
        setIsOffline(true);
      });
  }, []);

  const topProducts = edaData?.top_products && edaData.top_products.length > 0 ? edaData.top_products : REAL_TOP_PRODUCTS;
  const keyFindings = edaData?.key_findings && edaData.key_findings.length > 0 ? edaData.key_findings : REAL_KEY_FINDINGS;
  const maxPositive = Math.max(...FEEDBACK_TRENDS.map((t) => t.positive));

  return (
    <AppShell page="insights" onNavigate={onNavigate}>
      <PageHeader
        title="Customer Intelligence & EDA"
        subtitle="Business insights, catalog sentiment aggregation, rating discrepancies, and text patterns across 109,490 customer reviews."
        icon={<Lightbulb className="h-6 w-6" />}
        actions={
          isOffline ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-800">
              <span className="h-2 w-2 rounded-full bg-amber-500" />
              Offline Snapshot
            </span>
          ) : (
            <LiveBadge label="Empirical EDA Insights" />
          )
        }
      />

      {/* Top Topics Section */}
      <div className="grid gap-6 xl:grid-cols-2">
        <Card
          title="Top Customer Praise Drivers"
          subtitle="Key positive themes identified in customer reviews."
          badge={<LiveBadge label="Dataset Analyzed" />}
        >
          <TopicList items={POSITIVE_TOPICS} color="bg-emerald-500" icon="up" />
        </Card>

        <Card
          title="Common Customer Complaints"
          subtitle="Highest frequency issues detected in negative reviews."
          badge={<LiveBadge label="Actionable Feedback" />}
        >
          <TopicList items={COMMON_COMPLAINTS} color="bg-rose-500" icon="down" />
        </Card>
      </div>

      {/* Product Catalog Sentiment Aggregation */}
      <Card
        className="mt-6"
        title="Product-Level Sentiment Aggregation"
        subtitle="Review volume, average star ratings, and sentiment breakdown for top catalog items."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[650px]">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 uppercase tracking-wide bg-slate-50/50">
                <th className="py-3 px-3">Product Name</th>
                <th className="py-3 px-3 text-right">Total Reviews</th>
                <th className="py-3 px-3 text-right">Avg Rating</th>
                <th className="py-3 px-3 text-right">Positive %</th>
                <th className="py-3 px-3 text-right">Negative %</th>
                <th className="py-3 px-3">Sentiment Ratio</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {topProducts.map((prod) => (
                <tr key={prod.product_name} className="hover:bg-slate-50/70 transition-colors">
                  <td className="py-3 px-3 font-semibold text-slate-800 flex items-center gap-2">
                    <Package className="h-4 w-4 text-slate-400 shrink-0" />
                    <span className="truncate max-w-xs">{prod.product_name}</span>
                  </td>
                  <td className="py-3 px-3 text-right font-mono text-slate-600">
                    {prod.total_reviews.toLocaleString()}
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-slate-800 flex items-center justify-end gap-1">
                    <Star className="h-3 w-3 text-amber-500 fill-amber-500" />
                    {prod.average_rating.toFixed(2)}
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-emerald-600">
                    {prod.positive_pct.toFixed(0)}%
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-rose-600">
                    {prod.negative_pct.toFixed(0)}%
                  </td>
                  <td className="py-3 px-3">
                    <div className="flex h-2.5 w-32 overflow-hidden rounded-full bg-slate-100">
                      <div className="bg-emerald-500" style={{ width: `${prod.positive_pct}%` }} />
                      <div className="bg-amber-400" style={{ width: `${100 - prod.positive_pct - prod.negative_pct}%` }} />
                      <div className="bg-rose-500" style={{ width: `${prod.negative_pct}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Review Volume Trends & Discrepancies */}
      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card
          title="Review Volume & Sentiment Stability"
          subtitle="Monthly volume of positive vs. negative feedback over the catalog cycle."
        >
          <div className="flex h-52 items-end gap-3 sm:gap-5">
            {FEEDBACK_TRENDS.map((t) => (
              <div key={t.month} className="flex flex-1 flex-col items-center gap-2">
                <div className="flex h-full w-full items-end justify-center gap-1.5">
                  <div
                    className="w-full max-w-[20px] rounded-t bg-emerald-500 transition-all duration-500"
                    style={{ height: `${(t.positive / maxPositive) * 100}%` }}
                    title={`Positive: ${t.positive.toLocaleString()}`}
                  />
                  <div
                    className="w-full max-w-[20px] rounded-t bg-rose-400 transition-all duration-500"
                    style={{ height: `${(t.negative / maxPositive) * 100}%` }}
                    title={`Negative: ${t.negative.toLocaleString()}`}
                  />
                </div>
                <p className="text-[11px] font-medium text-slate-500">{t.month}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-center gap-6 text-xs font-medium text-slate-500">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" /> Positive
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm bg-rose-400" /> Negative
            </span>
          </div>
        </Card>

        {/* EDA Visual Plots Grid */}
        <Card
          title="Key Exploratory Data Analysis Charts"
          subtitle="Rating distribution and review length statistics."
        >
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-2 text-center">
              <p className="text-[11px] font-semibold text-slate-600 mb-1">Rating Distribution (1-5)</p>
              <img
                src="/reports/rating_distribution.png"
                alt="Rating distribution chart"
                className="max-h-36 mx-auto rounded shadow-sm object-contain"
              />
            </div>
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-2 text-center">
              <p className="text-[11px] font-semibold text-slate-600 mb-1">Review Length by Sentiment</p>
              <img
                src="/reports/review_length_distribution.png"
                alt="Review length chart"
                className="max-h-36 mx-auto rounded shadow-sm object-contain"
              />
            </div>
          </div>
          <p className="mt-3 text-[11px] text-slate-500 leading-relaxed">
            Negative reviews feature significantly longer text (dissatisfied users elaborate on specific defects) compared to positive reviews which often consist of terse praise.
          </p>
        </Card>
      </div>

      {/* Key Findings Card */}
      <Card
        className="mt-6"
        title="Key Findings & Academic Course Deliverables"
        subtitle="Empirical takeaways from data exploration and model development."
        badge={<LiveBadge label="6 Core Takeaways" />}
      >
        <div className="space-y-3">
          {keyFindings.map((finding, idx) => (
            <div key={idx} className="flex items-start gap-3 rounded-xl border border-slate-100 bg-slate-50/70 p-3.5 text-xs">
              <CheckCircle2 className="h-4 w-4 text-brand-600 shrink-0 mt-0.5" />
              <p className="text-slate-700 leading-relaxed">{finding}</p>
            </div>
          ))}
        </div>
      </Card>
    </AppShell>
  );
}
