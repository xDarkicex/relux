package dataset

import (
	"context"
	"io"
)

// prefetchResult is a batch or error delivered by the background
// goroutine.
type prefetchResult struct {
	batch Batch
	err   error
}

// PrefetchIterator wraps an Iterator with a background goroutine
// that pre-fetches the next batch while the training loop consumes
// the current one. bufferSize controls how many batches can be
// buffered ahead.
type PrefetchIterator struct {
	inner  Iterator
	ch     chan prefetchResult
	ctx    context.Context
	cancel context.CancelFunc
}

// NewPrefetchIterator creates a PrefetchIterator wrapping inner.
// bufferSize is the channel capacity (number of pre-fetched batches
// to hold). The background goroutine starts immediately.
func NewPrefetchIterator(inner Iterator, bufferSize int) *PrefetchIterator {
	if bufferSize <= 0 {
		bufferSize = 2
	}
	ctx, cancel := context.WithCancel(context.Background())
	p := &PrefetchIterator{
		inner:  inner,
		ch:     make(chan prefetchResult, bufferSize),
		ctx:    ctx,
		cancel: cancel,
	}
	go p.fetchLoop()
	return p
}

func (p *PrefetchIterator) fetchLoop() {
	for {
		batch, err := p.inner.Next()
		select {
		case p.ch <- prefetchResult{batch: batch, err: err}:
		case <-p.ctx.Done():
			return
		}
		// If EOF or error, stop fetching. Reset() will restart.
		if err != nil {
			return
		}
	}
}

// Next returns the next pre-fetched batch. Returns io.EOF when
// the inner iterator is exhausted.
func (p *PrefetchIterator) Next() (Batch, error) {
	r, ok := <-p.ch
	if !ok {
		return Batch{}, io.EOF
	}
	return r.batch, r.err
}

// Reset stops the background goroutine, drains pending batches,
// resets the inner iterator, and restarts prefetching.
func (p *PrefetchIterator) Reset() {
	p.cancel()
	// Drain remaining channel entries.
	for {
		select {
		case <-p.ch:
		default:
			goto drained
		}
	}
drained:
	p.inner.Reset()
	p.ctx, p.cancel = context.WithCancel(context.Background())
	go p.fetchLoop()
}
