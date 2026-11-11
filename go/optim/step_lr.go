package optim

type StepLR struct {
	opt       Optimizer
	stepSize  int
	gamma     float64
	lastEpoch int
}

func NewStepLR(opt Optimizer, stepSize int, gamma float64) *StepLR {
	return &StepLR{opt: opt, stepSize: stepSize, gamma: gamma}
}
func (s *StepLR) Step() {
	s.lastEpoch++
	if s.lastEpoch%s.stepSize == 0 {
		for _, g := range s.opt.ParamGroups() {
			g.LR *= s.gamma
		}
	}
}
