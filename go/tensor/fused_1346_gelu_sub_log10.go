package tensor

import "fmt"

func Fused1346GeluSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Log10()
	return r, nil
}
func Fused1346GeluSubLog10Name() string { return "gelu_sub_log10" }
