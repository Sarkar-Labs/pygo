package tensor

import "fmt"

func Fused0484SqrtMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Log()
	return r, nil
}
func Fused0484SqrtMatmulLogName() string { return "sqrt_matmul_log" }
