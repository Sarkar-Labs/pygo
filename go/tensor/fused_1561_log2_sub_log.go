package tensor

import "fmt"

func Fused1561Log2SubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Log()
	return r, nil
}
func Fused1561Log2SubLogName() string { return "log2_sub_log" }
