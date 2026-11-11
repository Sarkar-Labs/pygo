package tensor

import "fmt"

func Fused1146RoundSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.GELU()
	return r, nil
}
func Fused1146RoundSubGeluName() string { return "round_sub_gelu" }
