package bridge

import (
	"context"
	"sync"
)

// GeneratedPythonToGoWorker executes callbacks safely and propagates cancellation.
type GeneratedPythonToGoWorker struct {
	mu    sync.Mutex
	calls uint64
}

func (w *GeneratedPythonToGoWorker) Do(ctx context.Context, fn func(context.Context) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	w.mu.Lock()
	w.calls++
	w.mu.Unlock()
	if fn == nil {
		return nil
	}
	return fn(ctx)
}
func (w *GeneratedPythonToGoWorker) Calls() uint64 { w.mu.Lock(); defer w.mu.Unlock(); return w.calls }
