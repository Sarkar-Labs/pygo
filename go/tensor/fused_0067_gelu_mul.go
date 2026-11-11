package tensor

import "fmt"

func Fused0067GeluMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	return r, nil
}
func Fused0067GeluMulName() string { return "gelu_mul" }
