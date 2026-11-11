package tensor

import "fmt"

func Fused1302GeluMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Cos()
	return r, nil
}
func Fused1302GeluMulCosName() string { return "gelu_mul_cos" }
