package tensor

import "fmt"

func Fused0352TanhMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.Log2()
	return r, nil
}
func Fused0352TanhMulLog2Name() string { return "tanh_mul_log2" }
