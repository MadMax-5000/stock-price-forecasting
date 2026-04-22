"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, ChevronDown } from "lucide-react";
import { Stock } from "@/types";
import { cn } from "@/lib/utils";
import { Label } from "./ui/label";
import {
    AppleIcon,
    MicrosoftIcon,
    AmazonIcon,
    GoogleIcon,
    MetaIcon,
    NetflixIcon,
    NvidiaIcon,
    AMDIcon,
    TeslaIcon,
    SalesforceIcon,
    OracleIcon,
    AdobeIcon,
    IntelIcon,
    CiscoIcon,
    IBMIcon,
    QualcommIcon,
} from "@/lib/icons";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
    AAPL: AppleIcon,
    MSFT: MicrosoftIcon,
    AMZN: AmazonIcon,
    GOOGL: GoogleIcon,
    META: MetaIcon,
    NFLX: NetflixIcon,
    NVDA: NvidiaIcon,
    AMD: AMDIcon,
    TSLA: TeslaIcon,
    CRM: SalesforceIcon,
    ADBE: AdobeIcon,
    ORCL: OracleIcon,
    INTC: IntelIcon,
    CSCO: CiscoIcon,
    IBM: IBMIcon,
    QCOM: QualcommIcon,
};

function StockIcon({ symbol, className }: { symbol: string; className?: string }) {
    const IconComponent = iconMap[symbol];
    if (!IconComponent) return null;
    return <IconComponent className={cn("w-full h-full", className)} />;
}

interface StockPickerProps {
    stocks: Stock[];
    selectedStock: string;
    onSelect: (symbol: string) => void;
}

export function StockPicker({ stocks, selectedStock, onSelect }: StockPickerProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [search, setSearch] = useState("");
    const inputRef = useRef<HTMLInputElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const selected = stocks.find((s) => s.symbol === selectedStock);

    const filteredStocks = stocks.filter(
        (s) =>
            s.symbol.toLowerCase().includes(search.toLowerCase()) ||
            s.name.toLowerCase().includes(search.toLowerCase()) ||
            s.sector.toLowerCase().includes(search.toLowerCase())
    );

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    return (
        <div ref={containerRef} className="relative w-full">
            <Label>Sélectionner une action</Label>
            <motion.button
                onClick={() => {
                    setIsOpen(!isOpen);
                    if (!isOpen) setTimeout(() => inputRef.current?.focus(), 100);
                }}
                className={cn(
                    "mt-2 w-full flex items-center justify-between px-4 py-3 rounded-xl",
                    "border border-mist-border bg-frosted-veil backdrop-blur-sm",
                    "text-left transition-all duration-200",
                    "hover:border-stone-gray"
                )}
                style={{ color: "var(--warm-parchment)" }}
                whileTap={{ scale: 0.99 }}
            >
                <div className="flex items-center gap-3">
                    <div
                        className="w-10 h-10 rounded-lg flex items-center justify-center p-1.5"
                        style={{ backgroundColor: "var(--earth-gray)" }}
                    >
                        <StockIcon symbol={selectedStock} className="text-warm-parchment" />
                    </div>
                    <div>
                        <div className="font-medium" style={{ color: "var(--warm-parchment)" }}>
                            {selected?.symbol || "Sélectionner"}
                        </div>
                        <div className="text-xs" style={{ color: "var(--stone-gray)" }}>
                            {selected?.name}
                        </div>
                    </div>
                </div>
                <motion.div
                    animate={{ rotate: isOpen ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                >
                    <ChevronDown className="w-5 h-5" style={{ color: "var(--stone-gray)" }} />
                </motion.div>
            </motion.button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.15 }}
                        className="absolute z-50 w-full mt-2 py-2 rounded-xl border border-mist-border"
                        style={{
                            backgroundColor: "var(--deep-void)",
                            boxShadow: "0 10px 40px rgba(0,0,0,0.5)",
                        }}
                    >
                        <div className="px-3 pb-2">
                            <div className="relative">
                                <Search
                                    className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
                                    style={{ color: "var(--stone-gray)" }}
                                />
                                <input
                                    ref={inputRef}
                                    type="text"
                                    value={search}
                                    onChange={(e) => setSearch(e.target.value)}
                                    placeholder="Rechercher des actions..."
                                    className="w-full pl-10 pr-4 py-2 rounded-lg text-sm outline-none placeholder:text-stone-gray"
                                    style={{
                                        backgroundColor: "var(--earth-gray)",
                                        color: "var(--warm-parchment)",
                                    }}
                                />
                            </div>
                        </div>
                        <div className="max-h-64 overflow-y-auto">
                            {filteredStocks.map((stock) => (
                                <motion.button
                                    key={stock.symbol}
                                    onClick={() => {
                                        onSelect(stock.symbol);
                                        setIsOpen(false);
                                        setSearch("");
                                    }}
                                    className={cn(
                                        "w-full flex items-center gap-3 px-4 py-2.5 text-left",
                                        "transition-colors duration-150",
                                        stock.symbol === selectedStock
                                            ? "bg-frosted-veil"
                                            : "hover:bg-frosted-veil"
                                    )}
                                    whileHover={{ x: 4 }}
                                    whileTap={{ scale: 0.98 }}
                                >
                                    <div
                                        className="w-8 h-8 rounded-md flex items-center justify-center p-1"
                                        style={{ backgroundColor: "var(--earth-gray)" }}
                                    >
                                        <StockIcon symbol={stock.symbol} className="text-warm-parchment" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium text-sm" style={{ color: "var(--warm-parchment)" }}>
                                            {stock.symbol}
                                        </div>
                                        <div className="text-xs truncate" style={{ color: "var(--stone-gray)" }}>
                                            {stock.name}
                                        </div>
                                    </div>
                                    <div className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: "var(--frosted-veil)", color: "var(--ash-gray)" }}>
                                        {stock.sector}
                                    </div>
                                </motion.button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
