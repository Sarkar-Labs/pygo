package tensor

import "fmt"

func Fused0876CosSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.GELU()
	return r, nil
}
func Fused0876CosSubGeluName() string { return "cos_sub_gelu" }
