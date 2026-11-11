package tensor

import "fmt"

func Fused0356TanhSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Log10()
	return r, nil
}
func Fused0356TanhSubLog10Name() string { return "tanh_sub_log10" }
