import {
  ISeriesPrimitive,
  SeriesAttachedParameter,
  Time,
  IPrimitivePaneView,
  IPrimitivePaneRenderer,
  ISeriesApi,
  IChartApi,
  CandlestickData,
} from "lightweight-charts";
import { CanvasRenderingTarget2D, MediaCoordinatesRenderingScope } from "fancy-canvas";
import { AggregatedNews, SentimentSlice } from "./types";

type MarkerPosition = {
  x: number;
  y: number;
  radius: number;
  slices: SentimentSlice[];
  total: number;
  aboveBar: boolean;
};

const MARKER_PADDING = 8;
const MIN_RADIUS = 6;
const MAX_RADIUS = 14;

class NewsMarkerRenderer implements IPrimitivePaneRenderer {
  private _markers: MarkerPosition[] = [];

  update(markers: MarkerPosition[]) {
    this._markers = markers;
  }

  draw(target: CanvasRenderingTarget2D): void {
    target.useMediaCoordinateSpace((scope: MediaCoordinatesRenderingScope) => {
      const ctx = scope.context;

      for (const marker of this._markers) {
        const { x, y, radius, slices, total } = marker;

        if (slices.length === 1) {
          // Single color circle
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fillStyle = slices[0].color;
          ctx.fill();
        } else {
          // Split-color pie circle
          let startAngle = -Math.PI / 2; // Start from top
          for (const slice of slices) {
            const sweepAngle = (slice.count / total) * Math.PI * 2;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.arc(x, y, radius, startAngle, startAngle + sweepAngle);
            ctx.closePath();
            ctx.fillStyle = slice.color;
            ctx.fill();
            startAngle += sweepAngle;
          }
        }

        // Border
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  }
}

class NewsMarkerPaneView implements IPrimitivePaneView {
  private _renderer = new NewsMarkerRenderer();

  update(markers: MarkerPosition[]) {
    this._renderer.update(markers);
  }

  zOrder() {
    return "top" as const;
  }

  renderer(): IPrimitivePaneRenderer {
    return this._renderer;
  }
}

/**
 * Custom series primitive that renders split-color pie circles for news sentiment.
 * Each candle with news gets a single circle whose color slices represent
 * the distribution of bullish/bearish/neutral sentiment.
 */
export class NewsSentimentPrimitive implements ISeriesPrimitive<Time> {
  private _chart: IChartApi | null = null;
  private _series: ISeriesApi<"Candlestick"> | null = null;
  private _requestUpdate: (() => void) | null = null;
  private _paneView = new NewsMarkerPaneView();
  private _data: AggregatedNews[] = [];

  attached(param: SeriesAttachedParameter<Time, "Candlestick">) {
    this._chart = param.chart as unknown as IChartApi;
    this._series = param.series as unknown as ISeriesApi<"Candlestick">;
    this._requestUpdate = param.requestUpdate;
  }

  detached() {
    this._chart = null;
    this._series = null;
    this._requestUpdate = null;
  }

  setData(data: AggregatedNews[]) {
    this._data = data;
    this._requestUpdate?.();
  }

  updateAllViews() {
    const chart = this._chart;
    const series = this._series;
    if (!chart || !series) {
      this._paneView.update([]);
      return;
    }

    const timeScale = chart.timeScale();
    const markers: MarkerPosition[] = [];

    for (const agg of this._data) {
      const x = timeScale.timeToCoordinate(agg.time as Time);
      if (x === null) continue;

      // Find the candle bar to position above/below
      const bars = series.data() as CandlestickData<Time>[];
      const bar = findBar(bars, agg.time);
      if (!bar) continue;

      const aboveBar = agg.avgSentiment >= 0;
      const priceY = aboveBar
        ? series.priceToCoordinate(bar.high)
        : series.priceToCoordinate(bar.low);
      if (priceY === null) continue;

      const radius = markerRadius(agg.maxImpact);
      const y = aboveBar
        ? (priceY as number) - radius - MARKER_PADDING
        : (priceY as number) + radius + MARKER_PADDING;

      markers.push({
        x: x as number,
        y,
        radius,
        slices: agg.slices,
        total: agg.items.length,
        aboveBar,
      });
    }

    this._paneView.update(markers);
  }

  paneViews() {
    return [this._paneView];
  }
}

function markerRadius(impact: number): number {
  return MIN_RADIUS + ((impact - 1) / 2) * (MAX_RADIUS - MIN_RADIUS);
}

function findBar(bars: CandlestickData<Time>[], time: number): CandlestickData<Time> | null {
  // Binary search for matching bar
  let lo = 0;
  let hi = bars.length - 1;

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const barTime = bars[mid].time as number;
    if (barTime === time) return bars[mid];
    if (barTime < time) lo = mid + 1;
    else hi = mid - 1;
  }

  return null;
}
