package tensor

import "fmt"

func Fused0442AbsMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Log2()
	return r, nil
}
func Fused0442AbsMulLog2Name() string { return "abs_mul_log2" }
