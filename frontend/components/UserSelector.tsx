"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Film, AlertCircle } from "lucide-react";
import { fetchUsers, type UserInfo } from "@/lib/api";

interface Props {
  onSelect: (user: UserInfo) => void;
}

export default function UserSelector({ onSelect }: Props) {
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchUsers(20);
        setUsers(data.users);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load users");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 max-w-3xl">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="shimmer h-28 rounded-2xl" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card p-6 text-center max-w-md">
        <p className="text-[var(--accent-rose)] mb-2 flex items-center justify-center gap-2 text-sm">
          <AlertCircle className="h-4 w-4 shrink-0" strokeWidth={2} aria-hidden />
          {error}
        </p>
        <p className="text-sm text-[var(--text-muted)]">
          Point Rocky at the API — usually <code className="text-[var(--text-secondary)]">localhost:8000</code>
        </p>
      </div>
    );
  }

  const occupationColors: Record<string, string> = {
    student: "from-blue-500 to-cyan-400",
    engineer: "from-emerald-500 to-teal-400",
    programmer: "from-violet-500 to-purple-400",
    writer: "from-amber-500 to-orange-400",
    educator: "from-pink-500 to-rose-400",
    scientist: "from-indigo-500 to-blue-400",
    technician: "from-cyan-500 to-sky-400",
    administrator: "from-slate-500 to-gray-400",
    executive: "from-red-500 to-rose-400",
    other: "from-gray-500 to-slate-400",
  };

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 max-w-3xl">
      {users.map((user, i) => {
        const gradient = occupationColors[user.occupation] || occupationColors.other;
        return (
          <motion.button
            type="button"
            key={user.user_id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            onClick={() => onSelect(user)}
            className="glass-card p-4 text-left hover:scale-105 transition-transform duration-200 cursor-pointer group"
          >
            <div
              className={`w-10 h-10 rounded-full bg-gradient-to-br ${gradient} flex items-center justify-center text-white font-bold text-sm mb-3 group-hover:scale-110 transition-transform`}
            >
              {user.user_id}
            </div>
            <p className="text-sm font-medium text-[var(--text-primary)] truncate">
              User #{user.user_id}
            </p>
            <p className="text-xs text-[var(--text-muted)]">
              {user.age}y • {user.occupation}
            </p>
            <p className="text-xs text-[var(--text-secondary)] mt-1 flex items-center gap-1">
              <Film className="h-3 w-3 shrink-0 opacity-80" strokeWidth={2} aria-hidden />
              {user.num_movies_watched} logged
            </p>
          </motion.button>
        );
      })}
    </div>
  );
}
