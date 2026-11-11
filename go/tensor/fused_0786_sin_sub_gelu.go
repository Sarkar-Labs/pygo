package tensor

import "fmt"

func Fused0786SinSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sin()
	r = r.GELU()
	return r, nil
}
func Fused0786SinSubGeluName() string { return "sin_sub_gelu" }
