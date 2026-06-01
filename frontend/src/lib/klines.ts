import { Kline, Timeframe } from "./types";

const BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines";

const INTERVAL_MAP: Record<Timeframe, string> = {
  "1h": "1h",
  "4h": "4h",
  "1d": "1d",
  "1w": "1w",
  "1M": "1M",
};

// Default number of candles to load initially
const DEFAULT_CANDLE_COUNT = 200;

// Candle duration in ms
const CANDLE_MS: Record<Timeframe, number> = {
  "1h": 3600_000,
  "4h": 4 * 3600_000,
  "1d": 24 * 3600_000,
  "1w": 7 * 24 * 3600_000,
  "1M": 30 * 24 * 3600_000,
};

export async function fetchKlines(
  symbol: string,
  timeframe: Timeframe,
  startTime?: number,
  endTime?: number,
): Promise<Kline[]> {
  const now = Date.now();
  const interval = INTERVAL_MAP[timeframe];

  // Default: load DEFAULT_CANDLE_COUNT candles ending at now
  const effectiveEnd = endTime ?? now;
  const effectiveStart =
    startTime ?? effectiveEnd - DEFAULT_CANDLE_COUNT * CANDLE_MS[timeframe];

  const url = new URL(BINANCE_FAPI);
  url.searchParams.set("symbol", symbol.toUpperCase());
  url.searchParams.set("interval", interval);
  url.searchParams.set("startTime", String(effectiveStart));
  url.searchParams.set("endTime", String(effectiveEnd));
  url.searchParams.set("limit", String(DEFAULT_CANDLE_COUNT));

  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Binance API error: ${res.status}`);

  const data: number[][] = await res.json();

  return data.map((k) => ({
    time: Math.floor(k[0] / 1000),
    open: Number(k[1]),
    high: Number(k[2]),
    low: Number(k[3]),
    close: Number(k[4]),
  }));
}

/** Returns the candle duration in seconds for a given timeframe. */
export function candleDurationSec(timeframe: Timeframe): number {
  return CANDLE_MS[timeframe] / 1000;
}

/** User's UTC offset in seconds (e.g., UTC+3 → 10800). */
export function tzOffsetSec(): number {
  return new Date().getTimezoneOffset() * -60;
}
