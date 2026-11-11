package tensor

import "fmt"

func Fused0292TanhMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Sqrt()
	return r, nil
}
func Fused0292TanhMulSqrtName() string { return "tanh_mul_sqrt" }
