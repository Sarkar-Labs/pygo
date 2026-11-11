package tensor

import "fmt"

func Fused0516SqrtSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.GELU()
	return r, nil
}
func Fused0516SqrtSubGeluName() string { return "sqrt_sub_gelu" }
