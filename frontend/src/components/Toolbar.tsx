"use client";

import { Timeframe } from "../lib/types";

type Props = {
  ticker: string;
  timeframe: Timeframe;
  minImpact: number;
  onTickerChange: (t: string) => void;
  onTimeframeChange: (tf: Timeframe) => void;
  onImpactChange: (v: number) => void;
};

const TIMEFRAMES: Timeframe[] = ["1h", "4h", "1d", "1w", "1M"];

const IMPACT_LABELS: Record<1 | 2 | 3, string> = {
  1: "Notable",
  2: "High",
  3: "Extreme",
};

export default function Toolbar({
  ticker,
  timeframe,
  minImpact,
  onTickerChange,
  onTimeframeChange,
  onImpactChange,
}: Props) {
  return (
    <div className="flex items-center gap-4 px-4 py-2 border-b border-gray-200 bg-gray-50">
      {/* Ticker input */}
      <div className="flex items-center gap-1">
        <label className="text-xs text-gray-500 uppercase">Ticker</label>
        <input
          type="text"
          value={ticker}
          onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
          className="w-28 px-2 py-1 text-sm border border-gray-300 rounded bg-white focus:outline-none focus:border-blue-400"
        />
      </div>

      {/* Timeframe buttons */}
      <div className="flex items-center gap-1">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf}
            onClick={() => onTimeframeChange(tf)}
            className={`px-2 py-1 text-xs rounded ${
              tf === timeframe
                ? "bg-blue-500 text-white"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
            }`}
          >
            {tf}
          </button>
        ))}
      </div>

      {/* Impact filter */}
      <div className="flex items-center gap-1">
        <label className="text-xs text-gray-500 uppercase mr-1">Min Impact</label>
        {([1, 2, 3] as const).map((level) => (
          <button
            key={level}
            onClick={() => onImpactChange(level)}
            className={`px-2 py-1 text-xs rounded ${
              level === minImpact
                ? "bg-blue-500 text-white"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
            }`}
          >
            {IMPACT_LABELS[level]}
          </button>
        ))}
      </div>
    </div>
  );
}
