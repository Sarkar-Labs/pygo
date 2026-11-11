package tensor

import "fmt"

func Fused0606ExpSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Exp()
	r = r.GELU()
	return r, nil
}
func Fused0606ExpSubGeluName() string { return "exp_sub_gelu" }
