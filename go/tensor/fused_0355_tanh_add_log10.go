package tensor

import "fmt"

func Fused0355TanhAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Log10()
	return r, nil
}
func Fused0355TanhAddLog10Name() string { return "tanh_add_log10" }
