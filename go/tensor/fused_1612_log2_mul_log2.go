package tensor

import "fmt"

func Fused1612Log2MulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Log2()
	return r, nil
}
func Fused1612Log2MulLog2Name() string { return "log2_mul_log2" }
