package tensor

import "fmt"

func Fused0312TanhMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Cos()
	return r, nil
}
func Fused0312TanhMulCosName() string { return "tanh_mul_cos" }
