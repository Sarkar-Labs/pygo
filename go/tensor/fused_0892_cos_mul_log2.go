package tensor

import "fmt"

func Fused0892CosMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Log2()
	return r, nil
}
func Fused0892CosMulLog2Name() string { return "cos_mul_log2" }
