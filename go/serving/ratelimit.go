package serving

import (
	"net/http"
	"sync"
	"time"
)

type TokenBucket struct {
	mu                      sync.Mutex
	tokens, maxTokens, rate float64
	lastRefill              time.Time
}

func NewTokenBucket(max, rate float64) *TokenBucket {
	return &TokenBucket{tokens: max, maxTokens: max, rate: rate, lastRefill: time.Now()}
}
func (tb *TokenBucket) Allow(n float64) bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	el := time.Since(tb.lastRefill).Seconds()
	tb.lastRefill = time.Now()
	tb.tokens += el * tb.rate
	if tb.tokens > tb.maxTokens {
		tb.tokens = tb.maxTokens
	}
	if tb.tokens >= n {
		tb.tokens -= n
		return true
	}
	return false
}

type RateLimiter struct {
	mu           sync.RWMutex
	limiters     map[string]*TokenBucket
	limit, burst float64
}

func NewRateLimiter(limit, burst float64) *RateLimiter {
	return &RateLimiter{limiters: make(map[string]*TokenBucket), limit: limit, burst: burst}
}
func (rl *RateLimiter) get(key string) *TokenBucket {
	rl.mu.RLock()
	l, ok := rl.limiters[key]
	rl.mu.RUnlock()
	if ok {
		return l
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()
	l = NewTokenBucket(rl.burst, rl.limit)
	rl.limiters[key] = l
	return l
}
func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !rl.get(r.RemoteAddr).Allow(1) {
			http.Error(w, `{"error":"rate limited"}`, 429)
			return
		}
		next.ServeHTTP(w, r)
	})
}
