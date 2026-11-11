package tensor

import "fmt"

func Fused0844CosMatmulLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_log: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Log()
	return r, nil
}
func Fused0844CosMatmulLogName() string { return "cos_matmul_log" }
