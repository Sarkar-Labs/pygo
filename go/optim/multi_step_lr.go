package optim

type MultiStepLR struct {
	opt        Optimizer
	milestones []int
	gamma      float64
	step       int
}

func NewMultiStepLR(opt Optimizer, milestones []int, gamma float64) *MultiStepLR {
	return &MultiStepLR{opt: opt, milestones: milestones, gamma: gamma}
}
func (s *MultiStepLR) Step() {
	s.step++
	for _, m := range s.milestones {
		if s.step == m {
			for _, g := range s.opt.ParamGroups() {
				g.LR *= s.gamma
			}
		}
	}
}
