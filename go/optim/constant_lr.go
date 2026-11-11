package optim

type ConstantLR struct {
	opt              Optimizer
	factor           float64
	totalIters, step int
}

func NewConstantLR(opt Optimizer, factor float64, totalIters int) *ConstantLR {
	return &ConstantLR{opt: opt, factor: factor, totalIters: totalIters}
}
func (s *ConstantLR) Step() {
	s.step++
	if s.step <= s.totalIters {
		for _, g := range s.opt.ParamGroups() {
			g.LR *= s.factor
		}
	}
}
