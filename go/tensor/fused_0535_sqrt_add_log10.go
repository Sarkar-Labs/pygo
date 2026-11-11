package tensor

import "fmt"

func Fused0535SqrtAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Log10()
	return r, nil
}
func Fused0535SqrtAddLog10Name() string { return "sqrt_add_log10" }
