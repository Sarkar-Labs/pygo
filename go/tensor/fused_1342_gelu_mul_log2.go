package tensor

import "fmt"

func Fused1342GeluMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.Log2()
	return r, nil
}
func Fused1342GeluMulLog2Name() string { return "gelu_mul_log2" }
