package tensor

import "fmt"

func Fused0536SqrtSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Log10()
	return r, nil
}
func Fused0536SqrtSubLog10Name() string { return "sqrt_sub_log10" }
