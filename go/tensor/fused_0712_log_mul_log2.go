package tensor

import "fmt"

func Fused0712LogMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Log()
	r = r.Log2()
	return r, nil
}
func Fused0712LogMulLog2Name() string { return "log_mul_log2" }
