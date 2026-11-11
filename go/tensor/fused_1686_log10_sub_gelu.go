package tensor

import "fmt"

func Fused1686Log10SubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.GELU()
	return r, nil
}
func Fused1686Log10SubGeluName() string { return "log10_sub_gelu" }
