package tensor

import "fmt"

func Fused1416SiluSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.GELU()
	return r, nil
}
func Fused1416SiluSubGeluName() string { return "silu_sub_gelu" }
