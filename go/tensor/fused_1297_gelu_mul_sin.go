package tensor

import "fmt"

func Fused1297GeluMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Sin()
	return r, nil
}
func Fused1297GeluMulSinName() string { return "gelu_mul_sin" }
