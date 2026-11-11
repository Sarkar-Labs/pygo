package runtime

import (
	"context"
	"sync"
)

type Task struct {
	ID  string
	Fn  func() error
	Err error
}
type Executor struct {
	workers int
	taskCh  chan *Task
	wg      sync.WaitGroup
}

func NewExecutor(workers int) *Executor {
	e := &Executor{workers: workers, taskCh: make(chan *Task, workers*8)}
	for i := 0; i < workers; i++ {
		e.wg.Add(1)
		go e.worker()
	}
	return e
}
func (e *Executor) worker() {
	defer e.wg.Done()
	for t := range e.taskCh {
		t.Err = t.Fn()
	}
}
func (e *Executor) Submit(t *Task) { e.taskCh <- t }
func (e *Executor) Shutdown()      { close(e.taskCh); e.wg.Wait() }

func ParallelFor(ctx context.Context, n int, fn func(i int) error) []error {
	errs := make([]error, n)
	var wg sync.WaitGroup
	sem := make(chan struct{}, 8)
	for i := 0; i < n; i++ {
		i := i
		select {
		case <-ctx.Done():
			errs[i] = ctx.Err()
			continue
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func() { defer func() { <-sem; wg.Done() }(); errs[i] = fn(i) }()
	}
	wg.Wait()
	return errs
}
