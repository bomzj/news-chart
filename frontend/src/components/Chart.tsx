"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  createChart,
  CandlestickSeries,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  Time,
  LogicalRange,
  MouseEventParams,
} from "lightweight-charts";
import { Kline, Timeframe, NewsItem, AggregatedNews } from "../lib/types";
import { fetchKlines, candleDurationSec, tzOffsetSec } from "../lib/klines";
import { fetchNews } from "../lib/news";
import { aggregateNews } from "../lib/aggregate";
import { NewsSentimentPrimitive } from "../lib/news-markers";
import NewsPopup from "./NewsPopup";

type Props = {
  ticker: string;
  timeframe: Timeframe;
  minImpact: number;
};

const LIVE_REFRESH_INTERVAL_MS = 5 * 60 * 1000;

function chartData(klines: Kline[], offset: number): CandlestickData<Time>[] {
  return klines.map((k) => ({
    time: (k.time + offset) as Time,
    open: k.open,
    high: k.high,
    low: k.low,
    close: k.close,
  }));
}

function mergeCandleData(
  current: CandlestickData<Time>[],
  latest: CandlestickData<Time>[],
): CandlestickData<Time>[] {
  const byTime = new Map<number, CandlestickData<Time>>();

  for (const candle of [...current, ...latest]) {
    byTime.set(candle.time as number, candle);
  }

  return [...byTime.values()].sort(
    (left, right) => (left.time as number) - (right.time as number),
  );
}

function visibleBarTimes(
  chart: IChartApi,
  series: ISeriesApi<"Candlestick">,
): { from: number; to: number } | null {
  const range = chart.timeScale().getVisibleLogicalRange();
  const bars = series.data() as CandlestickData<Time>[];
  if (!range || bars.length === 0) return null;

  const fromIndex = Math.max(0, Math.floor(range.from));
  const toIndex = Math.min(bars.length - 1, Math.ceil(range.to));
  const from = bars[fromIndex]?.time;
  const to = bars[toIndex]?.time;

  if (typeof from !== "number" || typeof to !== "number") return null;
  return { from, to };
}

export default function Chart({ ticker, timeframe, minImpact }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const primitiveRef = useRef<NewsSentimentPrimitive | null>(null);
  const newsRef = useRef<NewsItem[]>([]);
  const loadedRangeRef = useRef<{ from: number; to: number } | null>(null);
  const [newsVersion, setNewsVersion] = useState(0);
  const [aggregated, setAggregated] = useState<AggregatedNews[]>([]);
  const [popup, setPopup] = useState<{
    x: number;
    y: number;
    news: AggregatedNews;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const fetchingOlderRef = useRef(false);
  const refreshingRef = useRef(false);
  const initialLoadDoneRef = useRef(false);
  const chartGenerationRef = useRef(0);

  const backendTicker = ticker.replace(/usdt$/i, "").toUpperCase();

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: "#ffffff" },
        textColor: "#1a1a1a",
      },
      grid: {
        vertLines: { color: "#e5e7eb" },
        horzLines: { color: "#e5e7eb" },
      },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
      },
      crosshair: {
        mode: 0,
      },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#16a34a",
      downColor: "#dc2626",
      borderUpColor: "#16a34a",
      borderDownColor: "#dc2626",
      wickUpColor: "#16a34a",
      wickDownColor: "#dc2626",
    });

    const markers = new NewsSentimentPrimitive();
    series.attachPrimitive(markers);

    chartRef.current = chart;
    seriesRef.current = series;
    primitiveRef.current = markers;

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      primitiveRef.current = null;
    };
  }, []);

  // Load klines when ticker/timeframe changes
  useEffect(() => {
    let cancelled = false;
    chartGenerationRef.current += 1;
    newsRef.current = [];
    loadedRangeRef.current = null;

    async function load() {
      setLoading(true);
      initialLoadDoneRef.current = false;
      try {
        const klines = await fetchKlines(ticker, timeframe);
        if (cancelled || !seriesRef.current) return;

        const data = chartData(klines, tzOffsetSec());

        seriesRef.current.setData(data);
        chartRef.current?.timeScale().fitContent();
        // Allow infinite scroll only after initial layout settles
        setTimeout(() => { initialLoadDoneRef.current = true; }, 500);
      } catch (err) {
        console.error("Failed to load klines:", err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [ticker, timeframe]);

  // Fetch news based on visible chart range (always fetches all impact levels)
  const fetchNewsForRange = useCallback(
    async (from: number, to: number) => {
      const generation = chartGenerationRef.current;
      try {
        const offset = tzOffsetSec();
        const candleDur = candleDurationSec(timeframe);
        const fromTs = new Date((from - offset) * 1000).toISOString();
        const toTs = new Date((to - offset + candleDur) * 1000).toISOString();
        const items = await fetchNews(backendTicker, fromTs, toTs, 1);
        if (generation !== chartGenerationRef.current) return;

        newsRef.current = items;
        loadedRangeRef.current = { from, to };
        setNewsVersion((v) => v + 1);
      } catch (err) {
        console.error("Failed to load news:", err);
      }
    },
    [backendTicker, timeframe],
  );

  // Keep the latest candle and its news current while the page remains open.
  const refreshChart = useCallback(async () => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series || refreshingRef.current) return;

    const generation = chartGenerationRef.current;
    const currentData = series.data() as CandlestickData<Time>[];
    const visibleRange = chart.timeScale().getVisibleLogicalRange();
    const followingLatest =
      visibleRange !== null &&
      currentData.length > 0 &&
      visibleRange.to >= currentData.length - 8;

    refreshingRef.current = true;
    try {
      const latest = chartData(await fetchKlines(ticker, timeframe), tzOffsetSec());
      if (
        generation !== chartGenerationRef.current ||
        !seriesRef.current ||
        !chartRef.current
      ) return;

      seriesRef.current.setData(mergeCandleData(currentData, latest));
      if (followingLatest) {
        chartRef.current.timeScale().scrollToRealTime();
      }

      const range = visibleBarTimes(chartRef.current, seriesRef.current);
      if (range) {
        await fetchNewsForRange(range.from, range.to);
      }
    } catch (err) {
      console.error("Failed to refresh chart:", err);
    } finally {
      refreshingRef.current = false;
    }
  }, [fetchNewsForRange, ticker, timeframe]);

  useEffect(() => {
    const refresh = () => { void refreshChart(); };
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") refresh();
    };

    const interval = window.setInterval(refresh, LIVE_REFRESH_INTERVAL_MS);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [refreshChart]);

  // Subscribe to visible range changes to fetch news
  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;

    let debounceTimeout: ReturnType<typeof setTimeout>;

    const handler = (range: LogicalRange | null) => {
      if (!range) return;
      clearTimeout(debounceTimeout);

      const bars = series.data();
      if (bars.length === 0) return;

      const fromIdx = Math.max(0, Math.floor(range.from));
      const toIdx = Math.min(bars.length - 1, Math.ceil(range.to));
      const fromBar = bars[fromIdx];
      const toBar = bars[toIdx];
      const fromTime = fromBar?.time as number;
      const toTime = toBar?.time as number;
      if (!fromTime || !toTime) return;

      // If visible range is within already-loaded range, skip fetch
      const loaded = loadedRangeRef.current;
      if (loaded && fromTime >= loaded.from && toTime <= loaded.to) return;

      // Debounce 3s before fetching news for newly exposed candles
      debounceTimeout = setTimeout(() => {
        fetchNewsForRange(fromTime, toTime);
      }, 3000);
    };

    chart.timeScale().subscribeVisibleLogicalRangeChange(handler);

    // Initial fetch after klines load
    const initialTimeout = setTimeout(() => {
      const logicalRange = chart.timeScale().getVisibleLogicalRange();
      if (logicalRange) handler(logicalRange);
    }, 1000);

    return () => {
      clearTimeout(debounceTimeout);
      clearTimeout(initialTimeout);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler);
    };
  }, [fetchNewsForRange]);

  // Infinite scroll: load older klines when user scrolls near left edge
  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;

    const handler = (range: LogicalRange | null) => {
      if (!range || fetchingOlderRef.current || !initialLoadDoneRef.current) return;

      // Trigger when user scrolls within 10 bars of the left edge
      if (range.from > 10) return;

      const bars = series.data() as CandlestickData<Time>[];
      if (bars.length === 0) return;

      const oldestTime = bars[0].time as number;
      const offset = tzOffsetSec();
      const oldestUtcMs = (oldestTime - offset) * 1000;

      fetchingOlderRef.current = true;
      fetchKlines(ticker, timeframe, undefined, oldestUtcMs - 1)
        .then((olderKlines) => {
          if (!seriesRef.current || olderKlines.length === 0) return;

          const olderData = chartData(olderKlines, offset);

          const currentData = seriesRef.current.data() as CandlestickData<Time>[];
          const merged = [...olderData, ...currentData];
          seriesRef.current.setData(merged);
        })
        .catch((err) => console.error("Failed to load older klines:", err))
        .finally(() => { fetchingOlderRef.current = false; });
    };

    chart.timeScale().subscribeVisibleLogicalRangeChange(handler);
    return () => chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler);
  }, [ticker, timeframe]);

  // Re-aggregate when minImpact changes or new news is fetched
  useEffect(() => {
    const series = seriesRef.current;
    const candleTimes = series
      ? (series.data() as CandlestickData<Time>[]).map((d) => d.time as number)
      : [];
    const agg = aggregateNews(newsRef.current, timeframe, minImpact, candleTimes);
    setAggregated(agg);
  }, [minImpact, timeframe, newsVersion]);

  // Update primitive with aggregated data
  useEffect(() => {
    if (!primitiveRef.current) return;
    primitiveRef.current.setData(aggregated);
  }, [aggregated]);

  // Handle click on chart to show news popup
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const handler = (param: MouseEventParams) => {
      if (!param.time || !param.point) {
        setPopup(null);
        return;
      }

      const clickedTime = param.time as number;
      const candleDur = candleDurationSec(timeframe);

      const match = aggregated.find(
        (agg) => Math.abs(agg.time - clickedTime) < candleDur,
      );

      if (match) {
        setPopup({ x: param.point.x, y: param.point.y, news: match });
      } else {
        setPopup(null);
      }
    };

    chart.subscribeClick(handler);
    return () => chart.unsubscribeClick(handler);
  }, [aggregated, timeframe]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/70">
          <span className="text-gray-500 text-sm">Loading chart...</span>
        </div>
      )}
      {popup && (
        <NewsPopup
          x={popup.x}
          y={popup.y}
          news={popup.news}
          onClose={() => setPopup(null)}
        />
      )}
    </div>
  );
}
