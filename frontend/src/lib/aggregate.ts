import { AggregatedNews, Impact, NewsItem, SentimentSlice, Timeframe } from "./types";
import { tzOffsetSec } from "./klines";

const BULLISH_COLOR = "#16a34a";
const BEARISH_COLOR = "#dc2626";

/**
 * Groups news items by the candle they fall into using actual chart candle times.
 * Binary-searches each news item into the correct candle interval.
 * Each candle gets one aggregated marker with averaged sentiment and max impact.
 */
export function aggregateNews(
  news: NewsItem[],
  timeframe: Timeframe,
  minImpact: number,
  candleTimes?: number[],
): AggregatedNews[] {
  if (!candleTimes || candleTimes.length === 0) return [];

  const offset = tzOffsetSec();
  const buckets = new Map<number, NewsItem[]>();

  for (const item of news) {
    if (item.impact < minImpact) continue;

    const publishedUtcSec = Math.floor(new Date(item.published_at).getTime() / 1000);
    const publishedChartTime = publishedUtcSec + offset;

    const candleTime = bsearchCandle(candleTimes, publishedChartTime);
    if (candleTime === null) continue;

    const existing = buckets.get(candleTime);
    if (existing) {
      existing.push(item);
    } else {
      buckets.set(candleTime, [item]);
    }
  }

  const aggregated: AggregatedNews[] = [];
  for (const [time, items] of buckets) {
    const avgSentiment = items.reduce((sum, i) => sum + (i.sentiment === "bullish" ? 1 : -1), 0) / items.length;
    const maxImpact = Math.max(...items.map((i) => i.impact)) as Impact;
    const slices = sentimentSlices(items);

    aggregated.push({ time, items, avgSentiment, maxImpact, slices });
  }

  return aggregated.sort((a, b) => a.time - b.time);
}

/** Computes pie slices for a set of news items grouped by sentiment direction. */
function sentimentSlices(items: NewsItem[]): SentimentSlice[] {
  let bullish = 0;
  let bearish = 0;

  for (const item of items) {
    if (item.sentiment === "bullish") bullish++;
    else bearish++;
  }

  const slices: SentimentSlice[] = [];
  if (bullish > 0) slices.push({ sentiment: "bullish", count: bullish, color: BULLISH_COLOR });
  if (bearish > 0) slices.push({ sentiment: "bearish", count: bearish, color: BEARISH_COLOR });

  return slices;
}

/** Binary search for the last candle time <= target. candleTimes must be sorted ascending. */
function bsearchCandle(candleTimes: number[], target: number): number | null {
  let lo = 0;
  let hi = candleTimes.length - 1;

  if (target < candleTimes[0]) return null;

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (candleTimes[mid] <= target) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  return candleTimes[hi];
}

/** Maps sentiment value to a color. Binary: bullish=green, bearish=red. */
export function sentimentColor(sentiment: string): string {
  return sentiment === "bullish" ? BULLISH_COLOR : BEARISH_COLOR;
}

/** Maps impact (1-3) to marker radius in pixels. */
export function impactRadius(impact: number): number {
  return 6 + impact * 4; // 10px to 18px
}
