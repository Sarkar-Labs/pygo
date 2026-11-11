package tensor

import "fmt"

func Fused0806SinSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.Log10()
	return r, nil
}
func Fused0806SinSubLog10Name() string { return "sin_sub_log10" }
