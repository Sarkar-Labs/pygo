package tensor

import "fmt"

func Fused0322TanhMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Ceil()
	return r, nil
}
func Fused0322TanhMulCeilName() string { return "tanh_mul_ceil" }
