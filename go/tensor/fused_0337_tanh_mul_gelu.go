package tensor

import "fmt"

func Fused0337TanhMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.GELU()
	return r, nil
}
func Fused0337TanhMulGeluName() string { return "tanh_mul_gelu" }
