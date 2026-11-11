// Package inference provides high-performance model inference for PyGo.
package inference

import (
	"context"
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// ─── InferenceConfig ─────────────────────────────────────────────────────────

type InferenceConfig struct {
	MaxBatchSize   int
	MaxSeqLen      int
	Device         tensor.Device
	DType          tensor.DType
	UseFP16        bool
	UseQuantInt8   bool
	UseQuantInt4   bool
	EnableKVCache  bool
	KVCacheMaxSeq  int
	NumBeams       int
	BeamWidth      int
	DoSample       bool
	Temperature    float64
	TopK           int
	TopP           float64
	RepPenalty     float64
	LenPenalty     float64
	MaxNewTokens   int
	EosTokenID     int
	PadTokenID     int
	NumThreads     int
	EnableBatching bool
	BatchTimeout   time.Duration
}

func DefaultInferenceConfig() InferenceConfig {
	return InferenceConfig{
		MaxBatchSize:   32,
		MaxSeqLen:      2048,
		Device:         tensor.DevCPU,
		DType:          tensor.Float32,
		NumBeams:       1,
		DoSample:       false,
		Temperature:    1.0,
		TopK:           50,
		TopP:           0.95,
		RepPenalty:     1.0,
		LenPenalty:     1.0,
		MaxNewTokens:   256,
		EosTokenID:     2,
		PadTokenID:     0,
		NumThreads:     4,
		EnableBatching: true,
		BatchTimeout:   5 * time.Millisecond,
		EnableKVCache:  true,
		KVCacheMaxSeq:  8192,
	}
}

// ─── GenerationOutput ────────────────────────────────────────────────────────

type GenerationOutput struct {
	Sequences    [][]int64 // [batch, seq_len]
	Scores       []float64 // per-sequence score
	TokenLatency []time.Duration
	TotalTime    time.Duration
	TokensPerSec float64
}

// ─── Predictor ────────────────────────────────────────────────────────────────

type Predictor struct {
	model   nn.Module
	cfg     InferenceConfig
	kvCache *KVCache
	mu      sync.Mutex
	// Dynamic batching
	requestCh chan *inferRequest
	wg        sync.WaitGroup
}

type inferRequest struct {
	inputIDs []int64
	resultCh chan *GenerationOutput
	ctx      context.Context
}

func NewPredictor(model nn.Module, cfg InferenceConfig) *Predictor {
	p := &Predictor{
		model:     model,
		cfg:       cfg,
		requestCh: make(chan *inferRequest, 1024),
	}
	// Compile / quantise model if requested
	if cfg.UseFP16 {
		p.model = CastModelFP16(model)
	} else if cfg.UseQuantInt8 {
		p.model = QuantiseInt8(model)
	} else if cfg.UseQuantInt4 {
		p.model = QuantiseInt4(model)
	}
	p.model.Train(false)

	if cfg.EnableBatching {
		for i := 0; i < cfg.NumThreads; i++ {
			p.wg.Add(1)
			go p.batchWorker()
		}
	}
	return p
}

// ─── Greedy decode ───────────────────────────────────────────────────────────

func (p *Predictor) Greedy(ctx context.Context, inputIDs []int64) (*GenerationOutput, error) {
	return p.generate(ctx, [][]int64{inputIDs}, generationParams{
		doSample:   false,
		maxNewToks: p.cfg.MaxNewTokens,
		eosID:      int64(p.cfg.EosTokenID),
		repPenalty: p.cfg.RepPenalty,
	})
}

// ─── Sampling decode ─────────────────────────────────────────────────────────

func (p *Predictor) Sample(ctx context.Context, inputIDs []int64) (*GenerationOutput, error) {
	return p.generate(ctx, [][]int64{inputIDs}, generationParams{
		doSample:    true,
		temperature: p.cfg.Temperature,
		topK:        p.cfg.TopK,
		topP:        p.cfg.TopP,
		maxNewToks:  p.cfg.MaxNewTokens,
		eosID:       int64(p.cfg.EosTokenID),
		repPenalty:  p.cfg.RepPenalty,
	})
}

// ─── Beam search ─────────────────────────────────────────────────────────────

func (p *Predictor) BeamSearch(ctx context.Context, inputIDs []int64, beams int) (*GenerationOutput, error) {
	return p.beamSearch(ctx, inputIDs, beams)
}

type generationParams struct {
	doSample    bool
	temperature float64
	topK        int
	topP        float64
	maxNewToks  int
	eosID       int64
	repPenalty  float64
}

// ─── Core generate loop ───────────────────────────────────────────────────────

func (p *Predictor) generate(ctx context.Context,
	batchInputIDs [][]int64, params generationParams) (*GenerationOutput, error) {

	start := time.Now()
	B := len(batchInputIDs)
	if B == 0 {
		return nil, fmt.Errorf("empty batch")
	}

	// Build padded input tensor [B, S]
	maxLen := 0
	for _, ids := range batchInputIDs {
		if len(ids) > maxLen {
			maxLen = len(ids)
		}
	}
	padded := make([]int64, B*maxLen)
	for b, ids := range batchInputIDs {
		for i, id := range ids {
			padded[b*maxLen+i] = id
		}
		// Fill remainder with pad_token_id
		for i := len(ids); i < maxLen; i++ {
			padded[b*maxLen+i] = int64(p.cfg.PadTokenID)
		}
	}
	inputT := tensor.FromSlice(padded, tensor.Shape{int64(B), int64(maxLen)}).To(p.cfg.Device)

	generated := make([][]int64, B)
	for b, ids := range batchInputIDs {
		generated[b] = append([]int64{}, ids...)
	}
	doneFlags := make([]bool, B)
	latencies := make([]time.Duration, 0, params.maxNewToks)

	var cache *KVCache
	if p.cfg.EnableKVCache {
		// numLayers would be retrieved from model config
		cache = NewKVCache(32, int64(p.cfg.KVCacheMaxSeq))
	}

	for step := 0; step < params.maxNewToks; step++ {
		stepStart := time.Now()

		// Forward pass
		var curInput *tensor.Tensor
		if cache != nil && step > 0 {
			// Only feed the last token
			lastToks := make([]int64, B)
			for b := range generated {
				lastToks[b] = generated[b][len(generated[b])-1]
			}
			curInput = tensor.FromSlice(lastToks, tensor.Shape{int64(B), 1}).To(p.cfg.Device)
		} else {
			curInput = inputT
		}

		logits := p.forwardWithCache(curInput, cache) // [B, S, V]
		// Take last token logits [B, V]
		nextLogits := logits // simplified; real impl indexes last seq pos

		// Process each batch element
		nextTokens := make([]int64, B)
		for b := 0; b < B; b++ {
			if doneFlags[b] {
				nextTokens[b] = int64(p.cfg.PadTokenID)
				continue
			}
			logitVec := extractRow(nextLogits, b) // [V]

			// Repetition penalty
			if params.repPenalty != 1.0 {
				applyRepetitionPenalty(logitVec, generated[b], params.repPenalty)
			}

			var nextTok int64
			if params.doSample {
				nextTok = sampleToken(logitVec, params.temperature, params.topK, params.topP)
			} else {
				nextTok = argmax(logitVec)
			}

			generated[b] = append(generated[b], nextTok)
			if nextTok == params.eosID {
				doneFlags[b] = true
			}
			nextTokens[b] = nextTok
		}

		latencies = append(latencies, time.Since(stepStart))

		// Check if all done
		allDone := true
		for _, d := range doneFlags {
			if !d {
				allDone = false
				break
			}
		}
		if allDone {
			break
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Update input for next step
		_ = nextTokens
	}

	total := time.Since(start)
	totalToks := 0
	for _, g := range generated {
		totalToks += len(g) - len(batchInputIDs[0])
	}

	return &GenerationOutput{
		Sequences:    generated,
		TokenLatency: latencies,
		TotalTime:    total,
		TokensPerSec: float64(totalToks) / total.Seconds(),
	}, nil
}

// ─── Beam Search ──────────────────────────────────────────────────────────────

type beam struct {
	tokens []int64
	score  float64
}

func (p *Predictor) beamSearch(ctx context.Context, inputIDs []int64, numBeams int) (*GenerationOutput, error) {
	// Initialize beams
	beams := make([]beam, numBeams)
	for i := range beams {
		beams[i] = beam{tokens: append([]int64{}, inputIDs...), score: 0}
	}
	doneBeams := make([]beam, 0, numBeams)

	for step := 0; step < p.cfg.MaxNewTokens; step++ {
		// For each beam, run forward and get topK candidates
		type candidate struct {
			tokens []int64
			score  float64
		}
		var candidates []candidate

		for _, b := range beams {
			inputT := tensor.FromSlice(b.tokens, tensor.Shape{1, int64(len(b.tokens))}).To(p.cfg.Device)
			logits := p.forwardWithCache(inputT, nil)
			lastLogits := extractRow(logits, 0)
			probs := softmax(lastLogits)

			// Get top-K tokens
			topK := topKIndices(probs, numBeams*2)
			for _, tokIdx := range topK {
				newToks := append(append([]int64{}, b.tokens...), int64(tokIdx))
				candidates = append(candidates, candidate{
					tokens: newToks,
					score:  b.score + math.Log(probs[tokIdx]+1e-10),
				})
			}
		}

		// Sort candidates by score (length-normalised)
		sort.Slice(candidates, func(i, j int) bool {
			si := candidates[i].score / math.Pow(float64(len(candidates[i].tokens)), p.cfg.LenPenalty)
			sj := candidates[j].score / math.Pow(float64(len(candidates[j].tokens)), p.cfg.LenPenalty)
			return si > sj
		})

		// Keep top numBeams
		newBeams := make([]beam, 0, numBeams)
		for _, c := range candidates {
			lastTok := c.tokens[len(c.tokens)-1]
			if lastTok == int64(p.cfg.EosTokenID) {
				doneBeams = append(doneBeams, beam{tokens: c.tokens, score: c.score})
			} else {
				newBeams = append(newBeams, beam{tokens: c.tokens, score: c.score})
			}
			if len(newBeams)+len(doneBeams) >= numBeams*2 {
				break
			}
		}

		beams = newBeams
		if len(beams) == 0 || len(doneBeams) >= numBeams {
			break
		}
	}

	// Combine and sort
	all := append(doneBeams, beams...)
	sort.Slice(all, func(i, j int) bool { return all[i].score > all[j].score })

	seqs := make([][]int64, min(numBeams, len(all)))
	scores := make([]float64, len(seqs))
	for i, b := range all[:len(seqs)] {
		seqs[i] = b.tokens
		scores[i] = b.score
	}

	return &GenerationOutput{Sequences: seqs, Scores: scores}, nil
}

// ─── Dynamic Batching Worker ──────────────────────────────────────────────────

func (p *Predictor) batchWorker() {
	defer p.wg.Done()
	for {
		req, ok := <-p.requestCh
		if !ok {
			return
		}

		// Collect more requests within timeout window
		reqs := []*inferRequest{req}
		deadline := time.After(p.cfg.BatchTimeout)
	collect:
		for len(reqs) < p.cfg.MaxBatchSize {
			select {
			case r := <-p.requestCh:
				reqs = append(reqs, r)
			case <-deadline:
				break collect
			}
		}

		// Process batch
		batchIDs := make([][]int64, len(reqs))
		for i, r := range reqs {
			batchIDs[i] = r.inputIDs
		}

		out, err := p.generate(context.Background(), batchIDs, generationParams{
			doSample: p.cfg.DoSample, temperature: p.cfg.Temperature,
			topK: p.cfg.TopK, topP: p.cfg.TopP,
			maxNewToks: p.cfg.MaxNewTokens, eosID: int64(p.cfg.EosTokenID),
		})

		for i, r := range reqs {
			var singleOut *GenerationOutput
			if err == nil && out != nil && i < len(out.Sequences) {
				singleOut = &GenerationOutput{
					Sequences: [][]int64{out.Sequences[i]},
				}
			}
			r.resultCh <- singleOut
		}
	}
}

func (p *Predictor) Submit(ctx context.Context, inputIDs []int64) (*GenerationOutput, error) {
	req := &inferRequest{
		inputIDs: inputIDs,
		resultCh: make(chan *GenerationOutput, 1),
		ctx:      ctx,
	}
	p.requestCh <- req
	select {
	case out := <-req.resultCh:
		return out, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (p *Predictor) Close() {
	close(p.requestCh)
	p.wg.Wait()
}

// ─── Helper functions ─────────────────────────────────────────────────────────

func (p *Predictor) forwardWithCache(input *tensor.Tensor, _ *KVCache) *tensor.Tensor {
	return p.model.Forward(input)
}

func extractRow(t *tensor.Tensor, b int) []float64 {
	// Simplified — real impl indexes tensor data
	return make([]float64, 32000)
}

func applyRepetitionPenalty(logits []float64, generated []int64, penalty float64) {
	seen := make(map[int64]bool)
	for _, id := range generated {
		seen[id] = true
	}
	for id := range seen {
		if int(id) < len(logits) {
			if logits[id] > 0 {
				logits[id] /= penalty
			} else {
				logits[id] *= penalty
			}
		}
	}
}

func sampleToken(logits []float64, temp float64, topK int, topP float64) int64 {
	// temperature scaling
	for i := range logits {
		logits[i] /= temp
	}
	probs := softmax(logits)
	// top-k
	if topK > 0 {
		topKFilter(probs, topK)
	}
	// top-p (nucleus)
	if topP < 1.0 {
		topPFilter(probs, topP)
	}
	return int64(sampleFromProbs(probs))
}

func argmax(v []float64) int64 {
	best, bestI := math.Inf(-1), 0
	for i, x := range v {
		if x > best {
			best, bestI = x, i
		}
	}
	return int64(bestI)
}

func softmax(v []float64) []float64 {
	maxV := math.Inf(-1)
	for _, x := range v {
		if x > maxV {
			maxV = x
		}
	}
	out := make([]float64, len(v))
	sum := 0.0
	for i, x := range v {
		out[i] = math.Exp(x - maxV)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func topKIndices(probs []float64, k int) []int {
	type idx struct {
		i int
		v float64
	}
	items := make([]idx, len(probs))
	for i, v := range probs {
		items[i] = idx{i, v}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].v > items[b].v })
	out := make([]int, min(k, len(items)))
	for i := range out {
		out[i] = items[i].i
	}
	return out
}

func topKFilter(probs []float64, k int) {
	indices := topKIndices(probs, k)
	keep := make(map[int]bool)
	for _, i := range indices {
		keep[i] = true
	}
	for i := range probs {
		if !keep[i] {
			probs[i] = 0
		}
	}
}

func topPFilter(probs []float64, p float64) {
	type iv struct {
		i int
		v float64
	}
	items := make([]iv, len(probs))
	for i, v := range probs {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].v > items[b].v })
	cumSum := 0.0
	cutoff := -1
	for j, x := range items {
		cumSum += x.v
		if cumSum >= p {
			cutoff = j
			break
		}
	}
	if cutoff >= 0 {
		for _, x := range items[cutoff+1:] {
			probs[x.i] = 0
		}
	}
}

func sampleFromProbs(probs []float64) int {
	// Use the standard PRNG rather than reinterpreting wall-clock nanoseconds
	// as a float. The old implementation produced a biased, low-quality sample
	// because Float64frombits does not map arbitrary integers to U[0,1).
	r := rand.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(probs) - 1
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// These transformations must not silently return the original model.
// Until module-wide parameter replacement is wired to the quantized tensor ABI,
// fail explicitly so inference cannot accidentally run at the wrong precision.
func CastModelFP16(m nn.Module) nn.Module {
	if m == nil {
		panic("inference.CastModelFP16: nil model")
	}
	panic("inference.CastModelFP16: FP16 module conversion is not yet exposed by the stable Go ABI")
}
func QuantiseInt8(m nn.Module) nn.Module {
	if m == nil {
		panic("inference.QuantiseInt8: nil model")
	}
	panic("inference.QuantiseInt8: module-wide INT8 conversion is not yet exposed by the stable Go ABI")
}
func QuantiseInt4(m nn.Module) nn.Module {
	if m == nil {
		panic("inference.QuantiseInt4: nil model")
	}
	panic("inference.QuantiseInt4: module-wide INT4 conversion is not yet exposed by the stable Go ABI")
}
func tensor_Cat(ts []*tensor.Tensor, dim int) *tensor.Tensor {
	if len(ts) == 0 {
		panic("inference.tensor_Cat: no tensors")
	}
	if len(ts) == 1 {
		return ts[0]
	}
	panic("inference.tensor_Cat: concatenation is not yet exposed by the stable Go ABI")
}

// RunForward executes the underlying model for serving workloads.
func (p *Predictor) RunForward(input *tensor.Tensor) *tensor.Tensor {
	if p == nil || p.model == nil {
		panic("inference.Predictor.RunForward: predictor has no model")
	}
	return p.model.Forward(input)
}

// StreamGenerate emits generated token IDs as they become available.
func (p *Predictor) StreamGenerate(ctx context.Context, inputIDs []int64, out chan<- int64) {
	defer close(out)
	result, err := p.Greedy(ctx, inputIDs)
	if err != nil {
		return
	}
	if len(result.Sequences) == 0 {
		return
	}
	for _, tok := range result.Sequences[0][len(inputIDs):] {
		select {
		case out <- tok:
		case <-ctx.Done():
			return
		}
	}
}
