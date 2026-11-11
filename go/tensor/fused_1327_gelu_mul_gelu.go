package tensor

import "fmt"

func Fused1327GeluMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.GELU()
	return r, nil
}
func Fused1327GeluMulGeluName() string { return "gelu_mul_gelu" }
